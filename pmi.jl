""" convert coom.jld to a csv of pmi scores
"""

using CSV
using DataFrames
using JSON
using JLD, HDF5
using SparseArrays
using ProgressBars

function main()

    # ------
    # script hard-coded variables
    # ------

    n = 15  # i.e., top n context words will be reported wrt., centre words

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

        # ------
        # container to hold, centre token + "_${tag}": top n pmi context tokens::Vector{String}
        # ------
        collated = Dict()  # init results dictionary

        # ------
        # load each co-occurence matrix present, corresponding to sub-corpi
        # separated by collection features (e.g., publisher)
        # ------

        # Vector{String} of file paths to jld files
        fns::Vector{String} =
            filter(x -> splitext(x)[2] == ".jld", readdir(output_dir))
        fps::Vector{String} = [joinpath(output_dir, fn) for fn in fns]


        # iterate over sub-corpora cooms, and collect ...
        for fp in fps

            vars = jldopen(fp, "r") do f
                read(f, "vars")
            end

            # sub-corpora tag
            tag::String = vars["tag"]

            # load the co-occurence (sparse) matrix
            cm::SparseMatrixCSC{Int64,Int64} = vars["cm"]

            # load the context word to row Dict
            w2i::Dict{String,Int64} = vars["w2i"]

            # load the centre word to column Dict
            c2i::Dict{String,Int64} = vars["c2i"]

            # load the log2(P(context)) Vector
            log_p_contexts::Vector{Float64} = vars["log_p_contexts"]

            # build index2word
            i2w = Dict([(v, k) for (k, v) in w2i])

            # ------
            # get the centre token : top n pmi context tokens::Vector{String}
            # ------
            for (centre_word, centre_i) in c2i

                # get a Vector of pmi score wrt., current centre word
                pmis::Vector{Float64} = get_pmis(centre_i, cm, log_p_contexts)

                # get Vector of (context word, pmi) tuples
                tuples = [(i2w[i], pmi) for (i, pmi) in enumerate(pmis)]

                # get a Vector of (centre word, pmi) tuples, ranked, highest 10 only
                ranked = sort(tuples, by = x -> x[2], rev = true)[1:n]

                # save
                collated[centre_word * "_$(tag)"] = ranked

            end

        end

        # ------
        # save the results to an inspectable csv
        # ------
        df = DataFrame(collated)

        # transpose the dataframe as per
        # https://stackoverflow.com/questions/37668312/transpose-of-julia-dataframe
        df[:, "id"] = 1:size(df,1)
        df2 = DataFrame([[names(df)]; collect.(eachrow(df))], [:column; Symbol.(axes(df, 1))])

        # write 
        csv_path = joinpath(output_dir, "pmi.csv")
        CSV.write(csv_path, df2)

    end

end


""" Return a Vector{Float64} of log2( p(context | centre) / p(context) ) wrt., centre word
    index, center_i. 

    The order is the same 
"""
function get_pmis(
    centre_i::Int64,
    cm::SparseMatrixCSC{Int64, Int64},
    log_p_contexts::Vector{Float64},
)

    # p(context | centre)
    log_p_context_centre =
        log2.(cm[:, centre_i]) .- log2(sum(cm[:, centre_i]))

    # pmi = log2( p(context | centre) / p(context) )
    pmis::Vector{Float64} = log_p_context_centre - log_p_contexts

    return pmis

end

main()
