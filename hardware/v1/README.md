# TACROSS v1 hardware

Fabrication files for the v1 16×16 tactile sensing board.

## Board

A 16×16 pressure-sensing array read out by an Arduino Nano:

| Part | Role |
| --- | --- |
| Arduino Nano | reads the array over serial, exposes the raw matrix |
| CD74HC4067 | 16-channel analog multiplexer, selects one row at a time |
| 2 × SN74HC595 | shift registers, drive the 16 column lines |

All parts are available from LCSC; supplier part numbers are in the BOM.

## Files

| File | Contents | Original name |
| --- | --- | --- |
| `tacross_v1_bom.xlsx` | Bill of materials: reference designators, values, LCSC part numbers | `BOM_开源16_16_第三版_PCB1_2_2026-10-15.xlsx` |
| `tacross_v1_gerber.zip` | Gerber fabrication package (send this to the PCB house as-is) | `Gerber_PCB1_2_2026-10-15.zip` |
| `tacross_v1_pick_and_place.xlsx` | Component centroids for assembly | `PickAndPlace_PCB1_2_2026_10_15.xlsx` |

Files were renamed to ASCII so the paths survive every archive and CI tool.
The original names are recorded above.

## Ordering

Hand `tacross_v1_gerber.zip` to the fabricator together with the board
parameters (layer count, thickness, surface finish) you used for the third
revision. For assembly, supply `tacross_v1_bom.xlsx` and
`tacross_v1_pick_and_place.xlsx`; the BOM already carries LCSC part numbers, so
the assembly house can quote directly from it.

## Status

Third revision of the v1 board, dated 2026-10-15. The files here are the ones
that were fabricated; they have not been re-checked against the schematic in
this repository, because the schematic is not included.

## Raw readings

The board outputs an uncalibrated 16×16 matrix. Turning it into force values
requires a per-taxel baseline and gain, which is what the `calibration/`
contracts described in [`docs/DATA_FORMAT.md`](../../docs/DATA_FORMAT.md) expect.
Do not report Newton-level accuracy without that calibration.
