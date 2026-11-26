# README Figures

These images are rendered from the paper's figure PDFs in
[Yiling-Ma/-ARR-Aug--AmbIdea](https://github.com/Yiling-Ma/-ARR-Aug--AmbIdea),
at commit `30519dda90aa33acbc46e82d36382be0834c2129`.

| Image | Paper source |
| --- | --- |
| `figures/overview.png` | `figures/case_study.pdf` |
| `figures/pipeline.png` | `figures/ideabench_pipline.pdf` |

Rendered with Poppler's `pdftoppm`, preserving the paper artwork:

```bash
pdftoppm -f 1 -singlefile -scale-to 2000 -png case_study.pdf overview
pdftoppm -f 1 -singlefile -scale-to 2400 -png ideabench_pipline.pdf pipeline
```
