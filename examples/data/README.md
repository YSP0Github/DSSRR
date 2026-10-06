# Sample data — Apollo lunar seismic records

Three 12-hour MiniSEED files are bundled here so that the examples and the test
suite can run against **real archival data** without any download step.

| File | Channel | Samples | Sampling rate |
|---|---|---|---|
| `XA.S12.01.MHZ.19760113_070152-19760113_190149.mseed` | XA.S12.01.MHZ | 286 187 | 6.625 Hz |
| `XA.S15.01.MHZ.19760113_070152-19760113_190149.mseed` | XA.S15.01.MHZ | 286 186 | 6.625 Hz |
| `XA.S16.01.MHZ.19760113_070152-19760113_190149.mseed` | XA.S16.01.MHZ | 286 187 | 6.625 Hz |

All three cover **1976-01-13, 07:01:52 UTC → 19:01:49 UTC**, are stored as
`int32` digital counts on a 10-bit grid (0 – 1023), and come from three
different Apollo landing sites:

| Station code | Mission |
|---|---|
| `S12` | Apollo 12 |
| `S15` | Apollo 15 |
| `S16` | Apollo 16 |

`MHZ` is the long-period vertical seismometer channel of the Apollo Passive
Seismic Experiment (PSE), which ran as part of the ALSEP arrays.

## Directory layout

There is **no required folder structure** — the scanner looks only at file
names, so a flat directory works just as well as a `YYYY/MM/` tree. Two file
naming conventions are understood:

| Kind | Pattern | Example |
|---|---|---|
| Day file | `NET.STA.YYYYMMDD.mseed` | `XA.S12.19760113.mseed` |
| Segment file | `NET.STA.LOC.CHA.START-END.mseed` | `XA.S12.01.MHZ.19760113_070152-19760113_190149.mseed` |

Day files carry no location/channel in the name, so the viewer falls back to
the `MHZ` channel when loading them. Segment files carry both, which is why
the GUI's **Stream** list shows the full `NET.STA.LOC.CHA` for those.

This directory is also a ready-made raw root for the batch pipeline:

```bash
dssrr-batch --raw-root examples/data --fix-root examples/data_repaired \
  --stations S12,S15,S16 --d0 19760101 --d1 19760229
```

## Provenance and licensing

Apollo lunar seismic data were produced by NASA's Apollo Lunar Surface
Experiments Package and are distributed openly through the
[IRIS / EarthScope Data Management Center](https://ds.iris.edu/ds/nodes/dmc/)
under network code `XA`. They are **US Government works and are not subject to
copyright in the United States**; the files are included here purely as small,
freely redistributable test fixtures.

If you use this data in published work, cite the original PSE data set rather
than this repository. The canonical DOI for the Apollo PSE data is maintained by
the IRIS DMC.

> These files are **not** installed with the `dssrr` Python package — they live
> in `examples/` only, so the wheel stays small. Nothing in the library requires
> them: `examples/01_quickstart.py` and the test suite's synthetic tests run
> without any data file at all.

## Regenerating / extending

To use your own data instead, point the examples at any directory containing
MiniSEED files and pass the sampling rate explicitly:

```bash
python examples/03_real_apollo_mseed.py --data-dir /path/to/mseed --sr 6.625
```
