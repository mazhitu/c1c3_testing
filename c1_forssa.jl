# I check the velocity, it's about 2.5km/s 0.3Hz-1Hz
# summer=4/1-9/30
# night=nleg=11:26

using Geodesics
using Dates
using HDF5
using FFTW
using SAC

include("./myfilters_module.jl")

const LOCATION_FILE="/Users/zma/DATA/locations.txt"
const FFT_DIR="/Users/zma/DATA/FFT/2012/"
const CCF_DIR="/Users/zma/DATA/CCF/2012_forSSA_correlate/"

const d1 = Date(2012,1,1)
const d2 = Date(2012,12,31)
const nseg = 47

const nt_wrong = 9112
const nt = 9113
const nt_original=18225

const downsamp_freq=5.0

const comp_s = ["BHZ" "BHE" "BHN"]
const comp_r = ["BHZ" "BHE" "BHN"]
const comp_a = ["BHZ" "BHE" "BHN"]

const stalist = ["STOR" "SP2" "GNW" "SQM" "RATT" "LON" "YACT" "TOLT" "HOOD" "BLOW" "LCCR" "FISH" "LEBA" "J33A" "J68A"]
# const stalist = ["STOR" "J33A" "J68A" "FISH" "LCCR" "GNW"]
# const stalist = ["J33A" "J43A" "J57A"]
# const stalist = ["GNW" "FISH" "TOLT" "SP2" "RATT" "STOR"]

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
    @. fft1=work1+im*work2
#    println(fft1[1:10],fft1[end-10:end])
    return log(sum(abs.(work1)))
end

# same as noise_module, n is the window half length
# for now, B[2]=mean(A[1]+A[2]+A[3]), if n=1; and B[1]=B[2]
# A and B are 1D Array
function mov_avg!(A,n,B)
    nlen::Int32=0
    sum::Float64=0.
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
end

function prepare(fft1,fft2,fft1abs,commonabs,c1fft)
    """ 
    prepare for the c1, which includes
    1. compute cross spectrum, conj(fft1).*conj(fft2)./fft1abs
    """
    @. c1fft=(conj(fft1)/abs(fft1))*(fft2/abs(fft2)) #correlate
    # @. c1fft=(conj(fft1)/fft1abs)*(fft2)
    # @. c1fft=(conj(fft1)/fft1abs)*(fft2/fft1abs)  #deconv
    if any(isnan,c1fft)
        println("NAN!")
        # for i=1:length(c1fft)
        #     println((i,c1fft[i],fft1[i],fft2[i],fft1abs[i]))
        # end
        exit()
    end
end

function runall()
    """ the main function that governs how you load the data """
    # allfiles=filter(x->x[end-2:end]==".h5",readdir(FFT_DIR))
    # allfiles=filter(x-> (x[end-2:end]==".h5" && 
    #   (any(sta->occursin(sta,x),stalist) ||
    #   x[1:3]=="UW.")),
    #   readdir(FFT_DIR))
    allfiles=filter(x-> (x[end-2:end]==".h5" && 
      (any(sta->occursin(sta,x),stalist))),readdir(FFT_DIR))
    nfiles=length(allfiles)
    println("we will be doing",nfiles,"files")


    # these are for loading the data
    rawfft=ones(ComplexF32,nt,nfiles*length(comp_a))
    work1=Array{Float32}(undef,nt_wrong)
    work2=Array{Float32}(undef,nt_wrong)
    filespaceid=HDF5.h5s_create_simple(2,[nseg,nt_wrong],C_NULL)
    memspaceid=HDF5.h5s_create_simple(1,[nt_wrong],C_NULL)

    # these are for prepare subroutine
    c1fft=Array{ComplexF32}(undef,nt,nfiles^2*length(comp_r)*length(comp_s))    
    absfft=Array{Float32}(undef,nt)
    absffts=zeros(Float32,nt,nfiles^2*length(comp_r)*length(comp_s))
    absfft_common=similar(absfft)
    fft1work=Array{ComplexF32}(undef,nt)

    # stores the index for the c1
    colindx=Dict{String,Int32}()
    newindx=0
    colnsta=Array{String}(undef,nfiles*length(comp_a))
    colpair=Array{String}(undef,nfiles^2*length(comp_r)*length(comp_s))
    c1stack=zeros(ComplexF32,nt,nfiles^2*length(comp_r)*length(comp_s))
    n1stack=zeros(Int32,nfiles^2*length(comp_r)*length(comp_s))

    ifftplan=plan_irfft(c1stack[:,1],nt_original)
    tr=SAC.sample()

    #calling names(h5file) is surprisingly slow
    h5files=Array{HDF5File}(undef,nfiles)
    namerefs=Array{String}(undef,nfiles)
    datasetnames=Array{Array{String,1},1}()
    for (idx,file) in enumerate(allfiles)
        h5files[idx]=h5open(FFT_DIR*file,"r")
        push!(datasetnames,names(h5files[idx]))
        namerefs[idx]=datasetnames[end][4][1:end-5]
    end

    for day=map(x->Dates.format(x,"yyyy_mm_dd"),d1:Dates.Day(1):d2)
       println("doing day:",day)

       for iseg=1:nseg
            # println("doing iseg:",iseg)

            # this bits load the raw fft into matrix
            t1=time()
            ndx=0
            for (datasetname,nameref,h5file) in zip(datasetnames,namerefs,h5files)
                tmp=split(nameref,"_")

                for comp in comp_a
                    name=join([tmp[1],tmp[2],tmp[3],comp,day],"_")
                    if name*".real" in datasetname

                        if read(attrs(h5file[name*".real"]),"std")[iseg]>=10
                            println("std large:",(day,iseg,name))
                            continue
                        end

                        ndx+=1
                        check_energy=loadHDF5_seg!(view(rawfft,1:nt_wrong,ndx),h5file,name,work1,work2,iseg,filespaceid,memspaceid)
                        if (check_energy<-9) || (check_energy>-3)
                            println(("wrong instrument!!!!",day,name,iseg))
                            ndx-=1
                            continue
                        end

                        # WARNING::!!!! the last element is not included!!!!!!
                        rawfft[nt,ndx]=abs(rawfft[nt-1,ndx])

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
            timeall[1]+=t2-t1

            if ndx<=1 continue end
            # for i=1:ndx
            #     println(i,colnsta[i])
            # end


            # now prepare the c1; also establish the index if this pair doesn't exist before
            t1=time()
            ncol=0
            for i=1:ndx
                mov_avg!(view(rawfft,:,i),10,absfft)
                for j=1:ndx
                    ncol+=1
                    prepare(view(rawfft,:,i),view(rawfft,:,j),absfft,absfft_common,view(c1fft,:,ncol))
                    col=colnsta[i]*"_"*colnsta[j]
                    colpair[ncol]=col
                    if (col in keys(colindx)) == false
                        newindx+=1
                        colindx[col]=newindx
                    end
                end
            end
            t2=time()
            timeall[2]+=t2-t1

            t1=time()
            # do the c1 stack for each segment
            for i=1:ncol     
                idx=colindx[colpair[i]]
                for k=1:nt
                    c1stack[k,idx]+=c1fft[k,i]
                end
                n1stack[idx]+=1
            end
            t2=time()
            timeall[3]+=t2-t1

        end  #finish all segments in this day

        #output sac files for this day
        t1=time()
        for key in keys(colindx)
            idx=colindx[key]
            tmp=split(key,"_")
            tr=SAC.sample()
            tr.delta=1.0/downsamp_freq
            tr.evla=locations[tmp[1]][1]
            tr.evlo=locations[tmp[1]][2]
            tr.stla=locations[tmp[3]][1]
            tr.stlo=locations[tmp[3]][2]
            tr.kstnm=tmp[1]*tmp[3]
            if (n1stack[idx]>0)
                tr.npts=nt_original
                tr.t=fftshift((ifftplan*c1stack[:,idx]))/n1stack[idx]
                write(tr,CCF_DIR*key*"_"*day*".C1.SAC",byteswap=false)
            end
        end
        # return c1stack,absffts,absfft_common
        c1stack.=0.
        n1stack.=0
        c1fft.=0.
        t2=time()
        timeall[4]+=t2-t1
        
    end
    # return c1stack

    # close all h5files and ends
    foreach(close,h5files)
    HDF5.h5s_close(memspaceid)
    HDF5.h5s_close(filespaceid)
end

# c1stack,absffts,absfft_commom=runall()
# a1,a2=runall()
runall()
# absfft,absffts,absfft_common,ndx=runall()
println(timeall)