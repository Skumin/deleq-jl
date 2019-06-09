function gen_init_pop_simple(NP, boxbounds)
    dm = size(boxbounds)[1]
    xi = zeros(Float64, NP, dm)
    for i = 1:NP
        boundsok = false
        while !boundsok
            num = rand(dm)
            num = num./sum(num)
            ids = falses(dm)
            for j = 1:dm
                if num[j] >= boxbounds[j, 1] && num[j] <= boxbounds[j, 2]
                    ids[j] = true
                end
            end
            if all(ids)
                xi[i, :] = num
                boundsok = true
            end
        end
    end
    return xi
end

function mutate(mat, boxbounds, fParam)
    sz = size(mat)
    newmat = Array{Float64}(undef, sz)
    sqn = collect(1:sz[1])
    for i = 1:sz[1]
        boundsok = false
        while !boundsok
            rs = rand(sqn[1:end .!= i], 3)
            num = mat[rs[1], :] + fParam * (mat[rs[2], :] - mat[rs[3], :])
            ids = falses(sz[2])
            for j = 1:sz[2]
                if num[j] >= boxbounds[j, 1] && num[j] <= boxbounds[j, 2]
                    ids[j] = true
                end
            end
            if all(ids)
                newmat[i, :] = num
                boundsok = true
            end
        end
    end
    return newmat
end

function crossover(mat, newmat, cr)
    return (1 - cr) * mat + cr * newmat
end

function project_population(mat, Emat, constr)
    sz = size(mat)
    Mmat = Emat * transpose(Emat)
    projmat = Array{Float64}(undef, sz)
    for i = 1:sz[1]
        num = mat[i, :]
        z = Emat * num - constr
        u = inv(Mmat) * z
        v = transpose(Emat) * u
        projmat[i, :] = num - v
    end
    return projmat
end

function run_deleq(fun, boxbounds, cr, fParam, maxgen, NP, showProgress, Emat, constr, args...)
    gen = 1
    mat = gen_init_pop_simple(NP, boxbounds)
    funvals = vec(mapslices(x -> fun(x, args...), mat, dims = 2))
    rbest = argmax(funvals)
    trugen = maxgen
    while gen <= trugen
        if showProgress
            println(string("Generation ", gen))
        end
        pbest = rbest
        pbest_ind = mat[pbest[1], :]
        pbest_val = fun(pbest_ind, args...)
        newmat = mutate(mat, boxbounds, fParam)
        newmat = crossover(mat, newmat, cr)
        funvals1 = vec(mapslices(x -> fun(x, args...), newmat, dims = 2))
        inds = funvals1 .> funvals
        mat[inds, :] = newmat[inds, :]
        funvals[inds] = funvals1[inds]
        rbest = argmax(funvals)
        if any(Emat * mat[rbest[1], :] .!= transpose(constr))
            if showProgress
                println("* Unfeasible, projecting back.")
            end
            trugen -= 1
            if gen <= trugen
                projmat = project_population(mat, Emat, constr)
                funvals1 = vec(mapslices(x -> fun(x, args...), projmat, dims = 2))
                rbest = argmax(funvals1)
                if pbest_val > fun(projmat[rbest[1], :], args...)
                    projmat[rbest[1], :] = pbest_ind
                    rbest = pbest
                end
                mat = projmat
            else
                mat[rbest[1], :] = pbest_ind
            end
        end
        gen += 1
        if showProgress
            println(string("** Value = ", funvals[rbest[1]]))
        end
    end
    return mat[rbest[1], :], fun(mat[rbest[1], :], args...)
end
