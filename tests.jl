using BenchmarkTools

include("src\\functions.jl");

function rastrigin(x)
    return 10 * length(x) + sum(x .^ 2 - 10 * cos.(2 * pi .* x))
end

function rastrigin_minus(x)
    return -1 * rastrigin(x)
end

boxbounds = [-5 5; -5 5];
Emat = [1 1];
constr = [0];
NP = 50;
maxgen = 200;
mutate_type = 1;

@benchmark run_deleq(rastrigin_minus, $boxbounds, mutate_type, 0.7, 0.9, $maxgen, $NP, false, $Emat, $constr)
