# I check the velocity, it's about 2.5km/s 0.3Hz-1Hz
# summer=4/1-9/30
# night=nleg=11:26

using Geodesics
using Dates
using HDF5
using Mmap
using FFTW
using Statistics
using SAC

include("./myfilters_module.jl")

const LOCATION_FILE="/Users/zma/DATA/locations.txt"
const FFT_DIR="/Users/zma/DATA/FFT/2012/"
const CCF_DIR="/Users/zma/DATA/CCF/2012_forSSA_C3correlate_BHENZ/"

const d1 = Date(2012,1,1)
const d2 = Date(2012,12,31)
# const d3 = Date(2012,10,1)
# const d4 = Date(2012,12,31)
const nseg = 47

const f1 = 0.01
const f2 = 0.40

#const nt_wrong = 36450
#const nt = 36451
#const nt_original=72900 

const nt_wrong = 9112
const nt = 9113
const nt_original=18225

# const nseg = 45
# const nt_wrong=18225
# const nt=18226
# const nt_original=36450

const downsamp_freq=5.0

const n1 = Int32(200*downsamp_freq)    #this is about 2*200km/(2km/s)*20
const nlen = Int32(1200*downsamp_freq) #window length for c3

#const n1 = Int32(50*downsamp_freq)    #this is about 2*200km/(2km/s)*20
#const nlen = Int32(200*downsamp_freq) #window length for c3

const n2 = n1+nlen-1
const n3 = div(nlen,2)+1

const comp_s = ["BHE" "BHN" "BHZ"]
const comp_r = ["BHZ"]
const comp_a = ["BHE" "BHN" "BHZ"]

#const stalist = ["STOR" "SP2" "GNW" "SQM" "RATT" "LON" "YACT" "TOLT" "HOOD" "BLOW" "LCCR" "FISH" "LEBA" "J33A" "J68A"]
# const stalist = ["STOR" "J33A" "J68A" "FISH" "LCCR" "GNW"]
const stalist = ["BLOW" "GNW" "STOR" "HOOD" "TOLT"]

# for fft1*fft2, ifft, fft, c3, load data
const timeall = [0.0 0.0 0.0 0.0 0.0]

# get the distances all at once, which may or may not be useful...
function readlatlon()
    """ for reasons I don't understand, this can be about 1km different from obspy """
    f=open(LOCATION_FILE)
    readline(f)
    locations=Dict()
    while !eof(f)
        tmp=split(readline(f),",")
        locations[tmp[2]]=(parse(Float32,tmp[3]),parse(Float32,tmp[4]))
    end

    distances=Dict()
    for sta1 in keys(locations)
        for sta2 in keys(locations)
            if sta1==sta2
                distances[sta1*"_"*sta2]=0.
            else
                distances[sta1*"_"*sta2]=Geodesics.surface_distance(
                    locations[sta1][2],locations[sta1][1],
                    locations[sta2][2],locations[sta2][1],6371.
                )
                distances[sta2*"_"*sta1]=distances[sta1*"_"*sta2]
            end
        end
    end
    locations,distances
end
const locations,distances=readlatlon()

# the std still needs to be improved
# also needs to use Kurama's complex version
function loadHDF5_seg_old!(fft1,dat,name,work1,work2,iseg)

    local nt=nt_wrong

    t1=time()
    obj=dat[name*".real"]
    prop=HDF5.h5d_get_access_plist(obj.id)
    ret=Ptr{Cint}[0]
    HDF5.h5f_get_vfd_handle(dat.id,prop,ret)
    io=unsafe_load(ret[1])
    offset=HDF5.h5d_get_offset(obj.id)+(iseg-1)*nt*sizeof(Float32)
    work1.=Mmap.mmap(fdio(io),Array{Float32,1},nt,offset)
    t2=time()
    println("in load",t2-t1)

    obj=dat[name*".imag"]
    prop=HDF5.h5d_get_access_plist(obj.id)
    ret=Ptr{Cint}[0]
    HDF5.h5f_get_vfd_handle(dat.id,prop,ret)
    io=unsafe_load(ret[1])
    offset=HDF5.h5d_get_offset(obj.id)+(iseg-1)*nt*sizeof(Float32)
    work2.=Mmap.mmap(fdio(io),Array{Float32,1},nt,offset)

    @. fft1=work1+1im*work2

    return
end

function loadHDF5_seg!(fft1,dat,name,work1,work2,iseg,filespaceid,memspaceid)
    #dim0 in those select_hyperslab goes to the right here...
    local nt=nt_wrong

    obj=dat[name*".real"]
    HDF5.h5s_select_hyperslab(filespaceid,HDF5.H5S_SELECT_SET,[iseg-1,0],[1,1],[1,nt],C_NULL)
    HDF5.h5d_read(obj.id,HDF5.hdf5_type_id(Float32),memspaceid,filespaceid,obj.xfer,work1)
    HDF5.refresh(obj)

    obj=dat[name*".imag"]
    HDF5.h5s_select_hyperslab(filespaceid,HDF5.H5S_SELECT_SET,[iseg-1,0],[1,1],[1,nt],C_NULL)
    HDF5.h5d_read(obj.id,HDF5.hdf5_type_id(Float32),memspaceid,filespaceid,obj.xfer,work2)
    HDF5.refresh(obj)

    @. fft1=work1+1im*work2
#    println(fft1[1:10],fft1[end-10:end])

    return
end

# same as noise_module, n is the window half length
# for now, B[2]=mean(A[1]+A[2]+A[3]), if n=1; and B[1]=B[2]
# A and B are 1D Array
function mov_avg!(A,n,B)
    nlen::Int32=0
    sum::Float32=0.
    n2::Int32=n*2+1
    if (length(A)<n2)
        println("error in mov_avg!!!")
        exit()
    end
    for i=1:n2-1
        sum+=abs(A[i])
    end
    i1::Int32=n2-1
    i2::Int32=1
    for i::Int32=n+1:length(A)-n
        i1+=1
        sum=sum+abs(A[i1])
        B[i]=sum/n2
        sum-=abs(A[i2])
        i2+=1
    end
    for i=1:n
        B[i]=abs(A[i])
    end
    for i=length(A)-n+1:length(A)
        B[i]=abs(A[i])
    end

    #for the stupid last element
    B[end]=1.0

end

function prepare(fft1,fft2,fft1abs,fftcommon,fftplan,ifftplan,n1,n2,c1fft,work1,work2,c1time)
    """ 
    prepare for the c3, which includes
    1. compute cross spectrum, conj(fft1).*conj(fft2)./fft1abs
    2. ifft to time domain
    3. cut the positive part from n1 for nlen samples
    4. fft back to the freq domain
    """
    t1=time()
    # @. work1=(conj(fft1)/fft1abs)*(fft2/fft1abs)
#    @. work1=conj(fft1)*fft2/fft1abs/fftcommon
#    @. work1=conj(fft1)*fft2/fft1abs.^2
    @. work1=(conj(fft1)/abs(fft1))*(fft2/abs(fft2))

    # freq_win=mygauss_genwin(f1,f2,downsamp_freq,length(work1))
    # @. work1=work1*freq_win


    if any(isnan,work1) 
        println("NAN!",abs.(fft1),abs.(fft2),work1)
        exit()
    end
    work1[1]=0.0
    t2=time()
    timeall[1]+=t2-t1

    t1=time()
    FFTW.mul!(work2,ifftplan,work1)
    t2=time()
    timeall[2]+=t2-t1

#    FFTW.mul!(c1fft,fftplan,work2[n1:n2])   #I don't know why this is not working

    t1=time()
    c1fft.=fftplan*work2[n1:n2]
    t2=time()
    timeall[3]+=t2-t1

    if (any(isnan,c1fft))
        println("c1 NAN!!!!!")
        exit()
    end

    c1time.=work2
end


function c3(colpair,colindx,ncol,c1fft,c3stack,n3stack)
    t1=time()
    for i=1:ncol
        tmp1=split(colpair[i],"_")
        if !(tmp1[2] in comp_s) continue end
        for j=i:ncol
            tmp2=split(colpair[j],"_")
            if (tmp2[1] != tmp1[1] ) break end   #the way we store C1 allows for early break
            if !(tmp2[2] in comp_r) continue end
            name=join([tmp1[3],tmp1[4],tmp2[3],tmp2[4]],"_")
#            println("doing:",(tmp1,tmp2,name))
            idx=colindx[name]        
            for k=1:n3
                c3stack[k,idx]+=conj(c1fft[k,i])*c1fft[k,j]
            end
            n3stack[idx]+=1
            if (any(isnan,c3stack[:,idx]))
                println("c3 NAN!!!! strange",idx,name)
                println(any(isnan,c1fft[:,i]))
                println(any(isnan,c1fft[:,j]))
                println(any(isnan,c1fft[:,i].*c1fft[:,j]))
                exit()
            end
        end
    end
    t2=time()
    timeall[4]+=t2-t1
end


function testc3()
    """ the main function that governs how you load the data """
#    allfiles=filter(x->x[end-2:end]==".h5",readdir(FFT_DIR))
    allfiles=filter(x-> x[end-2:end]==".h5" && any(sta->occursin(sta,x),stalist),readdir(FFT_DIR))
    nfiles=length(allfiles)
    println("we will be doing",nfiles,"files")


    # these are for loading the data
    rawfft=ones(ComplexF32,nt,nfiles*length(comp_a))
    work1=Array{Float32}(undef,nt_wrong)
    work2=Array{Float32}(undef,nt_wrong)
    filespaceid=HDF5.h5s_create_simple(2,[nseg,nt_wrong],C_NULL)
    memspaceid=HDF5.h5s_create_simple(1,[nt_wrong],C_NULL)


    # these are for prepare subroutine
    mulfft=Array{ComplexF32}(undef,nt,nfiles^2*length(comp_r)*length(comp_s))
    c1fft=Array{ComplexF32}(undef,div(nlen,2)+1,nfiles^2*length(comp_r)*length(comp_s))    
    absfft=Array{Float32}(undef,nt)
    absfft_common=similar(absfft)
    c1time=Array{Float32}(undef,nt_original,nfiles^2*length(comp_r)*length(comp_s))
    work3=Array{ComplexF32}(undef,nt)
    work4=Array{Float32}(undef,nt_original)
    ifftplan=plan_irfft(work3,nt_original)
    fftplan=plan_rfft(work4[n1:n2])


    # stores the index for the final c3 stack
    colindx=Dict{String,Int32}()
    newindx=0
    colnsta=Array{String}(undef,nfiles*length(comp_a))
    colpair=Array{String}(undef,nfiles^2*length(comp_s)*length(comp_r))
    c1stack=zeros(Float32,nt_original,nfiles^2*length(comp_r)*length(comp_s))
    n1stack=zeros(Int32,nfiles^2*length(comp_r)*length(comp_r))
    c3stack=zeros(ComplexF32,div(nlen,2)+1,nfiles^2*length(comp_r)*length(comp_s))
    n3stack=zeros(Int32,nfiles^2*length(comp_s)*length(comp_r)*length(comp_r))

    h5files=Array{HDF5File}(undef,nfiles)
    namerefs=Array{String}(undef,nfiles)
    datasetnames=Array{Array{String,1},1}()
    for (idx,file) in enumerate(allfiles)
        h5files[idx]=h5open(FFT_DIR*file,"r")
        push!(datasetnames,names(h5files[idx]))
        namerefs[idx]=datasetnames[end][4][1:end-5]
    end

    for day=map(x->Dates.format(x,"yyyy_mm_dd"),d1:Dates.Day(1):d2)
    # for day=map(x->Dates.format(x,"yyyy_mm_dd"),append!(collect(d1:Dates.Day(1):d2),collect(d3:Dates.Day(1):d4)))
        println("doing day:",day)
       for iseg=1:nseg

            # this bits load the raw fft into matrix
            t1=time()
            ndx=0
            for (datasetname,nameref,h5file) in zip(datasetnames,namerefs,h5files)
                tmp=split(nameref,"_")
                for comp in comp_a
                    name=join([tmp[1],tmp[2],tmp[3],comp,day],"_")
                    println("doing name: ",name)
                    if name*".real" in datasetname
                        if read(attrs(h5file[name*".real"]),"std")[iseg]>=10
#                            println("std large:",(day,iseg,name))
                            continue
                        end
                        ndx+=1
                        println("doing:",day," ",name," ",iseg)
                        loadHDF5_seg!(view(rawfft,1:nt_wrong,ndx),h5file,name,work1,work2,iseg,filespaceid,memspaceid)
                        println("get here")
                        if any(iszero,view(rawfft,1:nt_wrong,ndx))
                            println(("zeros!!!!",day,name,iseg))
                            ndx-=1
                            continue
                        end
                        colnsta[ndx]=tmp[3]*"_"*comp
                    end
                end
            end
            t2=time()
            timeall[5]+=t2-t1

            # if ndx<=3 continue end

            # now prepare the c1; also establish the index if this pair doesn't exist before
            ncol=0
            for i=1:ndx
                mov_avg!(view(rawfft,:,i),10,absfft)
                if (i==1)
                    absfft_common.=absfft
                end
                for j=1:ndx
                    ncol+=1
                    dist=distances[split(colnsta[i],"_")[1]*"_"*split(colnsta[j],"_")[1]]
#                    n1_local::Int32=Int32(floor(dist/1.0)*downsamp_freq)*2
#                    if (n1_local==0) n1_local=1 end
#                    n2_local::Int32=n1_local+nlen-1
#                    println((dist,n1_local,n2_local,n1,n2))

#                    prepare(view(rawfft,:,i),view(rawfft,:,j),absfft,absfft_common,fftplan,ifftplan,
#                        n1_local,n2_local,view(c1fft,:,ncol),work3,work4,view(c1time,:,ncol))

                    prepare(view(rawfft,:,i),view(rawfft,:,j),absfft,absfft_common,fftplan,ifftplan,
                        n1,n2,view(c1fft,:,ncol),work3,work4,view(c1time,:,ncol))


                    col=colnsta[i]*"_"*colnsta[j]
                    colpair[ncol]=col
                    if (col in keys(colindx)) == false
                        newindx+=1
                        colindx[col]=newindx
                    end
                end
            end

            # examine c1 stack just to be sure
            for i=1:ncol     
                idx=colindx[colpair[i]]
                for k=1:nt_original
                    c1stack[k,idx]+=c1time[k,i]
                end
                n1stack[idx]+=1
            end


            # let's do c3
            c3(colpair,colindx,ncol,c1fft,c3stack,n3stack)

        end
    end

    # close all h5files and ends
    foreach(close,h5files)
    HDF5.h5s_close(memspaceid)
    HDF5.h5s_close(filespaceid)
    for j=1:newindx
        for k=1:nt_original
            c1stack[k,j]/=n1stack[j]
        end
        for k=1:n3
            c3stack[k,j]/=n3stack[j]
        end
    end

    return c1stack,colindx,n1stack,c1fft,n3stack,c3stack

end

c1stack,cols,n1stack,c1fft,n3stack,c3stack=testc3()
#@time testc3()

for key in keys(cols)
    idx=cols[key]
    tmp=split(key,"_")
    tr=SAC.sample()
    tr.delta=1.0/downsamp_freq
    tr.evla=locations[tmp[1]][1]
    tr.evlo=locations[tmp[1]][2]
    tr.stla=locations[tmp[3]][1]
    tr.stlo=locations[tmp[3]][2]
    tr.kstnm=tmp[1]*tmp[3]

    #output c1
    if (n1stack[idx]>0)
        tr.npts=nt_original
        tr.t=fftshift(c1stack[:,idx])
        write(tr,CCF_DIR*key*".C1.SAC",byteswap=false)
    end

    #output c3
    if (n3stack[idx]>0)
        tr.npts=nlen
        tr.t=fftshift(irfft(c3stack[:,idx],nlen))
        if any(isnan,tr.t)
            println("found NAN!!!",key)
            exit()
        end
        write(tr,CCF_DIR*key*".C3.SAC",byteswap=false)
    end


    println(key,(n1stack[idx],n3stack[idx]))
end


println(timeall)