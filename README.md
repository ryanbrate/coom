# coom

For each configuration file in [coom_configs.json](coom_configs.json), assemble and save a co-occurrence matrices, and from those co-occurrence matrices assemble and save csv of top pmi context words wrt., centre words.

The corpus is assumed to be of the form (i.e., the output of github.com/ryanbrate/tokenize)

```
# A corpus
[
    [
        document label,
        [
            [], # A sentence, of token::String items
            ...
        ], 
    ], # A document
    ...
]
```

run:
```
julia -t 8 coom.jl
julia pmi.jl
```

## configuration options

### optional

    "name" : not used by the script

### necessary

    "input_dir"::str : location of collections of sentence segmented, tokenized documents
    "output_dir"::str : 
    "window"::int : size of context window, either side of centre to consider
    "centres"::list[str] : list of centre words to consider in building coom.
        All other centre words ignored
    "minFreq"::int : context words below this minimum frequency are ignored in the coom
    "filename_info_regexes"::list[str] : a list of regexes, specifying
        collection name attributes to extract. sub-corpora are considered with
        common regex extracted info, for which separated cooms are constructed,
        enabling pmi.jl to report centre word pmi statistics wrt., each
        sub-corpora (e.g., by date, by publisher).  If empty, the whole corpus is considered.
