# teach-by-showing

A visitor steers a simulated overhead crane whose load swings on a rope. From
that short demonstration a small learned model picks up, within seconds, how
to do the same job in the visitor's own style: cautious or brisk, depending on the
teacher. A demonstrator for an exhibition stand, not a library.

The point is the mechanism behind many industrial AI applications: people
teach by showing rather than by writing rules, and people decide what counts
as a good result.

**Status:** in development, for the KMI closing event on 3–4 November 2026.
It runs next to its sibling [feature-viz](https://github.com/foersterflorian/feature-viz),
one monitor each.

```bash
pdm install
pdm run pytest
pdm run teach-by-showing        # http://localhost:8081
```

The server runs the stand's steps (`DECISIONS.md` §18–§25): while it waits,
an agent that learnt from a person drives against one that learnt nothing;
A begins a demonstration of three transports; the agent learns from it,
shown in slow motion; then it drives the visitor's transports beside their
recorded drive, and targets it has never seen, while a style profile sets
the two drives side by side; then it joins the gallery and drives against
the agent most unlike it (§23). From the demonstration on, B returns the
stand to waiting.

The gallery keeps every agent under `agents/` in the data directory.
Ctrl+Alt+G at the stand shows only the agents taught from now on (see
Keyboard shortcuts); nothing is deleted. `teach-by-showing-gallery list`
shows them all, `teach-by-showing-gallery since all` (or `since #n`, or a
time stamp) sets the filter on the machine; in the container the same is
`docker compose exec teach-by-showing python -m teach_by_showing.gallery list`.

The screen speaks English or German (`DECISIONS.md` §24), English until
the staff switch with Ctrl+Alt+L, and dark or light (§25), dark until the
staff switch with Ctrl+Alt+H; both choices are kept in the data directory,
in `language.json` and `theme.json`, and survive a restart.

The crane is driven with the left stick of a USB gamepad, A and B are its
buttons; for development the keyboard stands in (see Keyboard shortcuts).
`--free` drives the crane without the stand's steps.
`http://localhost:8081/gamepad.html` shows what
the browser reads from a gamepad, and `tools/measure_ws.py` measures the
running server from outside — timing, input round trip (against `--free`),
and the command it applies (`DECISIONS.md` §8, §12):

```bash
pdm run python tools/measure_ws.py timing --seconds 30
```

On the stand it runs in a container, without GPU or devices; the browser on
the host draws the crane and reads the gamepad:

```bash
docker compose up -d                # then open http://localhost:8081
```

Each version's image is kept as an archive beside the repository, so that
the stand can be set up again without PyPI or Docker Hub (`DECISIONS.md`
§10):

```bash
docker save teach-by-showing:0.2.0 | zstd -T0 -o ../teach-by-showing-0.2.0.tar.zst
zstd -dc ../teach-by-showing-0.2.0.tar.zst | docker load
```

`tools/kiosk.py` opens the page full screen in Firefox on a chosen monitor,
picked by its EDID serial; it runs with the system Python on the host and
needs Firefox (`DECISIONS.md` §11):

```bash
python3 tools/kiosk.py list
python3 tools/kiosk.py open --serial <n>
```

`tools/udev/70-teach-by-showing-gamepad.rules` switches off the GameSir
gamepad's keyboard and mouse interface, through which its Share button
would open the desktop's screenshot dialog over the kiosk. It is installed
once on the host (`DECISIONS.md` §12):

```bash
sudo cp tools/udev/70-teach-by-showing-gamepad.rules /etc/udev/rules.d/
```

The GameSir runs in one of three modes, depending on what was pressed
(`DECISIONS.md` §12). The crane works in the default and the Xbox mode; the
stand relies on the default mode, which the pad falls back to on its own.
To see which mode it is in:

```bash
grep -i 'N: Name=.*gamesir' /proc/bus/input/devices
```

| Output | Mode | How it is reached |
|---|---|---|
| `GameSir-G7 SE Controller for Xbox` | default | plugged in, nothing pressed; LED off |
| `GameSir G7 SE` | Xbox | Menu + View + M held for 3 s while plugged in; LED on |
| `Guangzhou Chicken Run … GameSir-G7 SE Controller for Xbox` | View+Menu — wrong layout, replug | View and Menu held while plugging in; LED on |

Without a terminal, `gamepad.html` shows the same after a button press: the
id starts with `3537-1082-` (default), `3537-1010-GameSir G7 SE` (Xbox) or
`3537-1010-Guangzhou` (View+Menu). The LED alone does not tell the Xbox mode
from the View+Menu mode.

A teacher's drive is recorded as a run, and an agent cloned from it can be
measured and replayed beside it (`DECISIONS.md` §13–§17):

```bash
pdm run teach-by-showing --record vorsichtig      # the next 3 transports, to data/runs/
pdm run python tools/evaluate.py tests/data/*.json      # clone every run, measure the clones
pdm run teach-by-showing --ghost tests/data/<run>.json  # the clone drives beside its teacher
pdm run teach-by-showing --showcase tests/data/<run>.json  # the agent shown while the stand waits
```

`--params FILE` takes a JSON file with the parameters that differ from the
defaults: crane, task and input curve (`DECISIONS.md` §14, candidates in
`tools/tuning/`), and the durations of the stand's steps, for instance
`{"stand": {"learn_replay": 8, "learn_wait": 30}}` (`DECISIONS.md` §21).

## Keyboard shortcuts

Every key the stand's page reads, in the browser showing the crane:

| Keys | What they do |
|---|---|
| ← → or A / D | the stick, for development |
| Enter / Backspace | the gamepad's A / B, for development |
| Ctrl+Alt+G | staff: the gallery shows only the agents taught from now on (`DECISIONS.md` §23) |
| Ctrl+Alt+L | staff: the screen switches between English and German, and stays so (`DECISIONS.md` §24) |
| Ctrl+Alt+H | staff: the screen switches between dark and light, and stays so (`DECISIONS.md` §25) |

`DECISIONS.md` records why the code is shaped the way it is, including the
alternatives that were rejected. Read it before changing anything structural.

## Funding

The [KMI research and development project](https://kmi-netzwerk.org/kmi-projekt/) is funded as part of the “Future of Work: Regional Competence Centres for Labour Research – Artificial Intelligence” funding initiative within the “Innovations for Tomorrow's Production, Services, and Work” programme of the German Federal Ministry of Research, Technology and Space (BMFTR) and is supervised by the Project Management Agency Karlsruhe (PTKA).

<p align="left">
  <img src="src/teach_by_showing/funding/BMFTR_de_Web_RGB_gef_durch.jpg" alt="funding logo" width="300">
  <img src="src/teach_by_showing/funding/Logo_Kompetenzzentren_Arbeitsforschung.png" alt="logo ReKodA (Regional Competence Centres of Work Research)" width="150">
</p>

The running demonstrator carries the same notice on screen, or in German
the funder's own wording when the screen speaks German (`DECISIONS.md` §2,
§24).

## License

**MIT**. The full text is in [LICENSE](LICENSE). See `DECISIONS.md` §1.

The two logos in `src/teach_by_showing/funding/` are not covered by the MIT
licence. They belong to their respective owners and are included, unmodified,
solely to meet the funding acknowledgement requirement.

Dependencies are permissively licensed: NumPy, Starlette, uvicorn and
websockets (BSD-3-Clause), FastAPI and pydantic (MIT), and their own
dependencies (MIT, BSD-3-Clause, PSF-2.0).
