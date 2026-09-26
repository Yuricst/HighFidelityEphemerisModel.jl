# GVE versus Cartesian benchmark

Julia: `1.10.11`

## EOM evaluation

| Model | Minimum time (ns) | Allocations | Memory (bytes) |
|---|---:|---:|---:|
| Cartesian N-body | 6740 | 47 | 2576 |
| MEE N-body | 7025 | 53 | 3120 |
| Keplerian N-body | 12000 | 168 | 7168 |
| ordinary equinoctial N-body | 7475 | 56 | 3456 |
| Cartesian N-body+SH | 55700 | 1670 | 36592 |
| MEE N-body+SH | 56300 | 1676 | 37136 |
| Keplerian N-body+SH | 61100 | 1791 | 41184 |
| ordinary equinoctial N-body+SH | 56300 | 1679 | 37472 |

## RTN projection

| Projection | Minimum time (ns) | Allocations | Memory (bytes) |
|---|---:|---:|---:|
| Cartesian-derived | 16 | 0 | 0 |
| Direct MEE | 31 | 0 | 0 |

## Integrated propagation

| Model | Formulation | Initial condition | Cartesian time (s) | Element time (s) | Cartesian bytes | Element bytes | Final position error | Final velocity error | Max position error | Max velocity error |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N-body | MEE | general | 0.003140 | 0.002970 | 1139608 | 1119944 | 3.411142e-15 | 1.413083e-15 | 1.893416e-13 | 1.014839e-13 |
| N-body | MEE | near-circular planar | 0.002465 | 0.002021 | 934984 | 872888 | 2.001483e-15 | 7.021699e-16 | 5.242902e-13 | 2.843204e-13 |
| N-body | MEE | eccentric inclined | 0.001901 | 0.001780 | 730360 | 790536 | 4.965068e-16 | 1.241267e-16 | 4.485743e-13 | 1.348941e-13 |
| N-body | Keplerian | general | 0.019850 | 0.004310 | 1139608 | 2423352 | 3.024339e-15 | 1.518461e-15 | 2.484117e-13 | 1.188327e-13 |
| N-body | Keplerian | near-circular planar | 0.002489 | 0.007085 | 934984 | 3997064 | 1.100948e-12 | 3.416737e-13 | 1.279553e-12 | 3.805030e-13 |
| N-body | Keplerian | eccentric inclined | 0.001916 | 0.003302 | 730360 | 1933464 | 1.111610e-15 | 1.755417e-16 | 3.708510e-13 | 1.151022e-13 |
| N-body | ordinary equinoctial | general | 0.003021 | 0.002124 | 1139608 | 960920 | 3.066151e-15 | 1.413083e-15 | 4.630203e-13 | 2.179330e-13 |
| N-body | ordinary equinoctial | near-circular planar | 0.002448 | 0.001884 | 934984 | 869832 | 1.115760e-15 | 9.155201e-16 | 6.373731e-13 | 3.461733e-13 |
| N-body | ordinary equinoctial | eccentric inclined | 0.002040 | 0.002206 | 730360 | 869832 | 6.849500e-16 | 1.241267e-16 | 3.310996e-13 | 1.012083e-13 |
| N-body+SH | MEE | general | 0.025417 | 0.023228 | 15358296 | 12685384 | 2.925359e-15 | 1.899788e-15 | 1.965870e-13 | 1.035304e-13 |
| N-body+SH | MEE | near-circular planar | 0.030692 | 0.015669 | 12500424 | 9785080 | 2.391494e-15 | 4.577615e-16 | 4.922927e-13 | 2.668608e-13 |
| N-body+SH | MEE | eccentric inclined | 0.015077 | 0.013587 | 9642552 | 8818312 | 5.236912e-16 | 9.614813e-17 | 2.774638e-13 | 8.353550e-14 |
| N-body+SH | Keplerian | general | 0.024301 | 0.032936 | 15358296 | 14061704 | 4.419117e-15 | 2.303044e-15 | 2.426789e-13 | 1.202037e-13 |
| N-body+SH | Keplerian | near-circular planar | 0.019464 | 0.049193 | 12500424 | 23709848 | 1.100906e-12 | 3.417197e-13 | 1.277622e-12 | 3.805030e-13 |
| N-body+SH | Keplerian | eccentric inclined | 0.014996 | 0.014837 | 9642552 | 9773640 | 6.574030e-16 | 1.570092e-16 | 2.196594e-13 | 6.870749e-14 |
| N-body+SH | ordinary equinoctial | general | 0.025061 | 0.018490 | 15358296 | 11824120 | 1.678295e-15 | 1.408715e-15 | 2.861924e-13 | 1.641176e-13 |
| N-body+SH | ordinary equinoctial | near-circular planar | 0.020157 | 0.015326 | 12500424 | 9873112 | 1.790182e-15 | 8.082573e-16 | 3.893617e-13 | 2.108172e-13 |
| N-body+SH | ordinary equinoctial | eccentric inclined | 0.016857 | 0.025411 | 9642552 | 8897608 | 1.547358e-15 | 5.382006e-16 | 2.844821e-13 | 8.868319e-14 |
