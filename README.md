# teach-by-showing

A visitor steers a simulated overhead crane whose load swings on a rope. From
that short demonstration a small neural network learns, within seconds, to do
the same job in the visitor's own style: cautious or brisk, depending on the
teacher. A demonstrator for an exhibition stand, not a library.

The point is the mechanism behind many industrial AI applications: people
teach by showing rather than by writing rules, and people decide what counts
as a good result.

**Status:** in development, for the K-M-I closing event on 3–4 November 2026.
It runs next to its sibling [feature-viz](https://github.com/foersterflorian/feature-viz),
one monitor each.

```bash
pdm install
pdm run pytest
pdm run teach-by-showing        # http://localhost:8081
```

On the stand it runs in a container, without GPU or devices; the browser on
the host draws the crane and reads the gamepad:

```bash
docker compose up -d                # then open http://localhost:8081
```

`tools/kiosk.py` opens the page full screen on a chosen monitor, picked by
its EDID serial (`python3 tools/kiosk.py list`, then
`python3 tools/kiosk.py open --serial <n>`); see `DECISIONS.md` §11.

`DECISIONS.md` records why the code is shaped the way it is, including the
alternatives that were rejected. Read it before changing anything structural.

## Funding

The [K-M-I research and development project](https://kmi-netzwerk.org/kmi-projekt/) is funded as part of the “Future of Work: Regional Competence Centers for Labor Research – Artificial Intelligence” funding initiative within the “Innovations for Tomorrow's Production, Services, and Work” program of the German Federal Ministry of Research, Technology and Space (BMFTR) and is supervised by the Project Management Agency Karlsruhe (PTKA).

<p align="left">
  <img src="src/teach_by_showing/funding/BMFTR_de_Web_RGB_gef_durch.jpg" alt="funding logo" width="300">
  <img src="src/teach_by_showing/funding/Logo_Kompetenzzentren_Arbeitsforschung.png" alt="logo ReKodA (Regional Competence Centres of Work Research)" width="150">
</p>

The running demonstrator will carry the same notice on screen (`DECISIONS.md`
§2).

## License

**MIT**. The full text is in [LICENSE](LICENSE). See `DECISIONS.md` §1.

The two logos in `src/teach_by_showing/funding/` are not covered by the MIT
licence. They belong to their respective owners and are included, unmodified,
solely to meet the funding acknowledgement requirement.

Dependencies are permissively licensed: NumPy, Starlette, uvicorn and
websockets (BSD-3-Clause), FastAPI and pydantic (MIT).
