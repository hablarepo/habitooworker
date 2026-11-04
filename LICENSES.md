# Licence použitých knihoven

**Souhrn pro komerční použití:** všechny přímé závislosti workeru mají permisivní licence
(Apache-2.0, BSD, MIT) nebo LGPL pro dynamicky přibalené knihovny, které povolují komerční použití.
**To není právní porada** – před komerčním vydáním nech licence zkontrolovat (zejména body označené ⚠).
Ověřeno = přečteno z metadat/licenčního souboru balíčku v době psaní (listopad 2026);
„z paměti" = neověřeno touto session.

| Knihovna | Použití | Licence | Komerčně OK? | Ověřeno |
|----------|---------|---------|--------------|---------|
| **gsplat** 1.5.3 | trénink Gaussian splatu | Apache-2.0 | ano | ano (LICENSE v sdist) |
| **COLMAP / pycolmap** 4.2.1 | polohy kamer | BSD-3-Clause | ano | ano (metadata wheelu) |
| **PyTorch** 2.4.1 | tenzory, CUDA | BSD-3-Clause (styl) | ano | z paměti |
| **OpenCV** (opencv-python-headless) | čtení videa, rozmazání, zmenšení | Apache-2.0; wheel přibaluje **FFmpeg pod LGPL** | ano ⚠ | ano |
| **NumPy** | pole | BSD-3-Clause (+ permisivní součásti) | ano | ano |
| **supabase-py** | fronta a úložiště | MIT | ano | ano |
| **python-dotenv** | načtení `.env` | BSD-3-Clause | ano | ano |
| **pytest** (jen vývoj) | testy | MIT | ano (není v produkci) | ano |
| **Formát SPZ** (Niantic) | komprimovaný export | MIT (referenční repozitář) | ano | ano (LICENSE repozitáře) |
| **Základní obraz `nvidia/cuda`** | CUDA 12.1 v Dockeru | NVIDIA CUDA EULA / licence kontejneru | ano, pro běh vlastní aplikace ⚠ | z paměti |

## Poznámky

- **gsplat vs. původní 3DGS (Inria):** původní referenční implementace Inria má licenci jen pro
  nekomerční výzkum. Worker ji **nepoužívá**; gsplat je nezávislá implementace pod Apache-2.0.
- **HLOC (nepoužito):** Apache-2.0 kód, ale výchozí modely (SuperPoint/SuperGlue) mají podle mé paměti
  nekomerční licence vah – proto je v projektu vynecháno a polohy se počítají klasickým SIFT v COLMAPu.
  *(z paměti, před případným použitím ověř)*
- **SIFT:** patent vypršel (2020), COLMAP ho používá bez omezení.
- **SPZ:** worker obsahuje vlastní zapisovač podle zdrojů Niantic (MIT) – neinstaluje balíček `spz`
  z PyPI. Ten není od Niantic, ale cizí implementace.
- ⚠ **OpenCV / FFmpeg (LGPL):** wheel obsahuje FFmpeg jako sdílené knihovny (nenalezeny GPL kodeky
  x264/x265). LGPL dovoluje komerční použití při dynamickém linkování a zachování licenčních textů –
  ty jsou součástí wheelu (`LICENSE-3RD-PARTY.txt`). Pokud bys obraz šířil dál, zachovej je.
- ⚠ **pycolmap:** wheel přibaluje jen runtime GCC (libgomp, libgfortran, libquadmath – výjimka GCC
  Runtime Library). Licence staticky slinkovaných závislostí COLMAPu (např. SuiteSparse přes Ceres)
  jsem **neověřil**; před komerčním šířením obrazu to nech zkontrolovat.
- ⚠ **Obraz:** při sdílení hotového Docker obrazu třetím stranám platí licence všech balíčků
  v obrazu (včetně Ubuntu a NVIDIA CUDA). Pro vlastní provoz workeru (tvůj PC / tvá cloudová GPU) není problém.
