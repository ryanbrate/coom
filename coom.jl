""" Build a co-occurence matrix, and associated variables for only those centre words specified in each config.  Build co-occurence matrix for each confi in coom_configs.json.

run:
```
julia -t 8 coom_build.jl
```
"""

using ProgressBars: ProgressBar
using FromFile: @from
using DataStructures
using JLD, HDF5
using JSON
using ProgressBars
using SparseArrays


function main()

    # ------
    # load the configurations
    # ------
    configs::Vector = JSON.parsefile("coom_configs.json")

    # iterate over each config
    for (i, config) in enumerate(configs)

        println("\n\nrunning for config $(i)\n\n")

        # ------
        # load the config variables
        # ------
        output_dir::String = expanduser(config["output_dir"])
        input_dir::String = expanduser(config["input_dir"])
        centre_words::Vector{String} = config["centres"]  # centre words of interest

        minFreq::Int64 = config["minFreq"]
        window::Int64 = config["window"]
        filename_info_regexes::Vector{String} = config["filename_info_regexes"]

        #------
        # get the fps for collections in input_dir (ignoring any instances of config.json)
        #------
        fns::Vector{String} =
            filter(x -> splitext(x)[2] == ".json" && x != "config.json", readdir(input_dir))
        fps::Vector{String} = [joinpath(input_dir, fn) for fn in fns]

        # ------
        # sort the fps into groups according to config["filename_info_regexes"]
        # ------

        # if config["filename_info_regexes"] == [], then take whole corpus
        if length(filename_info_regexes) == 0
            fps_groups = Dict([(Tuple(["ALL"]), fps)])
            print(keys(fps_groups))
        else
            # identify unique, co-occurent info combinations wrt., collections
            info_combinations::Set{Vector} =
                Set([get_info(fp, filename_info_regexes) for fp in fps])

            # init a Dict of info combination=>[] for each info combinations
            fps_groups = Dict([(Tuple(subset), []) for subset in info_combinations])

            # assign fp instances
            for fp in fps
                subset_properties::Vector{String} = get_info(fp, filename_info_regexes)
                push!(fps_groups[Tuple(subset_properties)], fp)
            end
        end


        # ------
        # Assemble coom, and associated variables, associated with each group
        # ------

        # if output_dir exists ... skip
        if isdir(output_dir) == true

            println("$(output_dir) exists ... skipping")

        else

            # iterate over subset
            for (subset_properties::Tuple, subset_fps::Vector{String}) in fps_groups

                subset_tag::String = join(subset_properties, "_")

                # ------
                # get the corpus subset properties
                # ------
                println(
                    "\tconsidering corpus subset: $(subset_tag), a total of $(length(subset_fps)) collections",
                )

                println("\t\tassemble a counter of all tokens in the subset")
                count::DataStructures.Accumulator{String,Int64} = get_count(subset_fps)

                println("\t\tassemble centre token to index hash, wrt., coom")
                c2i = Dict([(cw, i) for (i, cw) in enumerate(centre_words)])

                println(
                    "\t\tassemble context token to index hash, wrt., coom, for only those tokens more frequent than minFreq",
                )
                w2i::Dict{String,Int64} = get_w2i(count, minFreq)

                # for the same order of tokens in w2i, create a Vector of log(P(token))
                println("\t\tassemble a Vector of log(P(token)) values")
                total::Int64 = sum(count)
                i2w = Dict([(i, w) for (w, i) in w2i])
                log_p_contexts::Vector{Float64} =
                    [log2(count[i2w[i]]) - log2(total) for i = 1:length(i2w)]

                # get a co-occurence matrix for the current config
                println("\t\tassemble the coom") 
                cm = getCoom(subset_fps, c2i, w2i, window)

                # ------
                # save the coom variables
                # ------

                println("\t\tsave")
                mkpath(output_dir)
                coom_path::String = joinpath(output_dir, "coom_$(subset_tag).jld")

                jldopen(coom_path, "w") do f
                    g = create_group(f, "vars")
                    g["cm"] = cm
                    g["w2i"] = w2i
                    g["c2i"] = c2i
                    g["log_p_contexts"] = log_p_contexts
                    g["tag"] = subset_tag
                end

                # ------
                # save config
                # ------
                open(joinpath(output_dir, "config.json"), "w") do f
                    JSON.print(f, config, 4)
                end


            end

        end
    end

end


""" Return co-occurence matrix::SparseMatrixCSC{Int64, Int64}

    Note: w2i has only those tokens whose frequency (corresponding to the
    corpus subset of fps) exceeds the specified config min frequency

    Args:

        fps::Vector{String} : A vector of file paths to corpus subset collections

        c2i::Dict : a Dict of centre words of interest to corresponding coom
                    column index

        w2i::Dict : a Dict of context words (with a frequency >=
                    config["minFreq"]) to corresponding coom row index

        window::Int64 : word window size either side of centre word taken as
                        context
"""
function getCoom(
    fps::Vector{String},
    c2i::Dict{String, Int64},
    w2i::Dict{String, Int64},
    window::Int64,
)

    # initialise the co-occurence matrix
    cm = spzeros(Int64, length(w2i), length(c2i))

    # centre words of interest
    centresOfInterest::Vector{String} = [c for (c, i) in c2i]

    # iterate over collection file paths and build co-occurence matrix
    for fp in ProgressBar(fps)

        # load the file
        collection = JSON.parsefile(fp)

        # iterate over doc in collection
        for (label, doc) in collection

            # consider each sentence in-turn
            for sentence::Vector{String} in doc

                # iterate over centre words
                for (i, centre::String) in enumerate(sentence)  # i is centre word index in toks

                    # only consider current centre word, if it's a centreOfInterest
                    if centre in centresOfInterest

                        # get context word set wrt., current centre word
                        contextWords::Set{String} = Set([
                            sentence[j] for
                            j = max(i - window, 1):min(i + window, length(sentence)) if
                            j != i
                        ])

                        # iterate over set of context words, and amend cm
                        for contextWord::String in contextWords
                            if contextWord in keys(w2i)  # i.e., if corpus subset freq > minFreq
                                cm[w2i[contextWord], c2i[centre]] += 1
                            end
                        end

                    end

                end

            end
        end

    end

    return cm

end

""" Return a token=>index Dict{String, Int64} wrt., counter, ignoring tokens in
    counter with frequency less than minFreq.

    Args:
        count::DataStructures.Accumulator{String, Int64}
        minFreq::Int64
"""
function get_w2i(count::DataStructures.Accumulator{String,Int64}, minFreq::Int64)

    uniqueTokens::Set = Set([token for (token, freq) in count if freq >= minFreq])

    return Dict([(token, i) for (i, token) in enumerate(uniqueTokens)])

end


""" Return a DataStructures.Accumulator{String, Int64} of token=>count for 
    corpus/sub-corpus represented by fps

    Args:
        fps::Vector{String}
"""
function get_count(fps::Vector{String})

    # container to hold counters for each collection
    counts = Vector{DataStructures.Accumulator{String,Int64}}(undef, length(fps))

    # iterate over collections, get counters wrt., each individ collection and
    # add to counts container
    Threads.@threads for i in ProgressBar(1:length(fps))
        counts[i] = get_collection_count(fps[i])
    end

    # merge counters into a single one, representing all collections
    return Dict(reduce(merge, counts))

end


""" Return a DataStructures.Accumulator{String, Int64} wrt., collection at file_path

    Args:
        file_path::String : the absolute filepath of the collection
"""
function get_collection_count(fp::String)

    # load the collection
    collection = JSON.parsefile(fp)

    # container to hold counters for each doc in the collection
    counts = Vector{DataStructures.Accumulator{String,Int64}}(undef, length(collection))

    # iterate over doc in collection, get counters wrt., each doc, add to container
    for (i, (label, doc)) in enumerate(collection)

        # each doc is a list of token lists, hence reduce to a single list
        all_doc_tokens::Vector{String} = reduce(vcat, doc)

        # get counter wrt., doc
        counts[i] = DataStructures.counter(all_doc_tokens)

    end

    # merge counters into a single one, representing the entire collection
    return reduce(merge, counts)

end


""" Return a Vector{String} of information from collection filename,
    according to specified regexes

    Args:
        fp::String : path to collection 
        filename_info_regexes::Vector{String} : Vector of regex Strings
"""
function get_info(fp::String, filename_info_regexes::Vector{String})

    fn::String = splitext(basename(fp))[1]

    # match each regex 
    info = String[]
    for regex in filename_info_regexes
        m = match(Regex(regex), fn)[1]
        push!(info, m)
    end

    return info

end

main()
