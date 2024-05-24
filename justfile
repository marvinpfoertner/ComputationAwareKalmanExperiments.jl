default:
    just --list

format:
    julia --project=@JuliaFormatter --eval 'using JuliaFormatter; format(".")'
