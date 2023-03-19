using NPZ
#using Plots
using GLMakie
using Telepathy
using Statistics

include("FastICA.jl")

npz = npzread("scikit_fastICA_BSS/sin_sign_saw.npz")["arr_0"]
#display(plot(1:2000, npz))
ica_result = fastICA(npz,  3)
#display(plot(1:2000, ica_result))

eegraw = Telepathy.load_data("data/P17_HN1.bdf")
#eegraw.data

ch_first, ch_last = get_channels(eegraw, "C5"), get_channels(eegraw, "C20")
fragm = eegraw.data[1:10000, ch_first:ch_last]

fig = Figure(resolution=(1000, 800))
ax = Axis(fig[1,1]) # remember about [1,1] after fig, otherwise it turns out too small
for chn in axes(fragm, 2)
   lines!(ax, fragm[:, chn] .- mean(fragm[:, chn]) .- 100*chn, color=:black, linewidth=0.5)
end
save("P17_HN1_C5-C20.png", fig)

fig = Figure(resolution=(1000, 800))
ax = Axis(fig[1,1])
eeg_ica = fastICA(fragm, 16) * 1000
for chn in axes(eeg_ica, 2)
    lines!(ax, eeg_ica[:, chn] .- mean(eeg_ica[:, chn]) .- 100*chn, linewidth=0.5, color=:black)
end
save("P17_HN1_C5-C20_fastICA.png", fig)


display(fig)