# courtesy of D. Hervas

using SolidStateDetectors # >= 5f3a5cb
using LegendDataManagement # >= ad9555c
using Plots
using Unitful
using ProgressMeter
using LegendHDF5IO
using HDF5
using Base.Threads

SSD = SolidStateDetectors
T = Float32

# let's use V10437B. Note: private metadata
meta = LegendDataManagement.readlprops("legend-metadata/hardware/detectors/germanium/diodes/V10437B.yaml")
xtal_meta = LegendDataManagement.readlprops("legend-metadata/hardware/detectors/germanium/crystals/V10437.yaml")
sim = Simulation{T}(LegendData, meta, xtal_meta)

sim.detector = SolidStateDetector(
    sim.detector,
    contact_id=2,
    contact_potential=meta.characterization.l200_site.recommended_voltage_in_V
)

@info "Calculating electric potential..."
calculate_electric_potential!(
    sim,
    refinement_limits=[0.2, 0.1, 0.05, 0.01],
    depletion_handling=true
)

@info "Calculating electric field..."
calculate_electric_field!(sim, n_points_in_φ=72)

@info "Calculating weighting potential..."
calculate_weighting_potential!(
    sim,
    sim.detector.contacts[1].id,
    refinement_limits=[0.2, 0.1, 0.05, 0.01],
    n_points_in_φ=2,
    verbose=false
)

function make_axis(boundary, gridsize)
    # exclude the exact endpoints to ensure all points are strictly inside the domain
    start = 0 + eps()
    stop = boundary - eps()

    # compute the number of intervals based on desired spacing
    n = round(Int, (stop - start) / gridsize)

    # adjust step to evenly divide the range
    step = (stop - start) / n

    # construct and return the axis as a range
    return range(start, step=step, length=n + 1)
end

gridsize = 0.001 # in m
radius = meta.geometry.radius_in_mm / 1000
height = meta.geometry.height_in_mm / 1000

x_axis = make_axis(radius, gridsize)
z_axis = make_axis(height, gridsize)

spawn_positions = CartesianPoint{T}[]
idx_spawn_positions = CartesianIndex[]
for (i,x) in enumerate(x_axis)
    for (k,z) in enumerate(z_axis)
        push!(spawn_positions, CartesianPoint(T[x,0,z]))
        push!(idx_spawn_positions, CartesianIndex(i,k))
    end
end
in_idx = findall(x -> x in sim.detector && !in(x, sim.detector.contacts), spawn_positions)

# simulate events

time_step = T(1)u"ns"
max_nsteps = 10000

# prepare thread-local storage
n = length(in_idx)
wfs_raw_threaded = Vector{Vector{Float64}}(undef, n)
dt_threaded = Vector{Int}(undef, n)

@info "Simulating energy depositions in r=$x_axis z=$z_axis grid..."
@threads for i in 1:n
    p = spawn_positions[in_idx[i]]
    e = SSD.Event([p], [2039u"keV"])
    simulate!(e, sim, Δt = time_step, max_nsteps = max_nsteps, verbose = false)

    # store results in preallocated arrays
    wf = add_baseline_and_extend_tail(e.waveforms[1], 2000, 7000).signal
    wfs_raw_threaded[i] = ustrip(wf)
    dt_threaded[i] = length(e.waveforms[1].signal)
end

# assign final results
wfs_raw = wfs_raw_threaded
dt = dt_threaded

wfs = wfs_raw ./ maximum.(wfs_raw)

drift_time = fill(NaN, length(x_axis), length(z_axis))
for (i, idx) in enumerate(idx_spawn_positions[in_idx])
    drift_time[idx] = dt[i]
end

output = (
    r=collect(x_axis) * u"m",
    z=collect(z_axis) * u"m",
    drift_time=transpose(drift_time) * u"ns"
)

@info "Saving to disk..."
lh5open("drift-time-maps.lh5", "w") do f
    f["V99999A"] = output
end
