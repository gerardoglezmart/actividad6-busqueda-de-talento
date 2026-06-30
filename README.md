# Sistema Inteligente para Búsqueda de Talento (LLM + datos simulados)

**Actividad 6 — Semana 9 · Procesamiento de Lenguaje Natural · MNA, Tecnológico de Monterrey**

* **Autor:** Gerardo González Martínez · **Matrícula:** A01840096
* **Profesor:** Luis Eduardo Falcón Morales

---

## 1. Resumen

Las organizaciones reciben grandes volúmenes de currículums y, al abrir una vacante especializada, la búsqueda
manual del mejor perfil es lenta y costosa. Este proyecto implementa un **sistema multimodal de búsqueda de talento**
que, a partir de una **vacante real**, analiza una base de conocimiento de **20 CVs sintéticos** (en formatos **PDF y
PNG**) y **recupera, jerarquiza y justifica** a los **5 mejores candidatos**.

El sistema sigue un enfoque **RAG (Retrieval-Augmented Generation)** aplicado a talento: vectoriza los CVs en una base
de datos vectorial, realiza **búsqueda semántica** de la vacante contra esa base, aplica un **ranking híbrido**
(explicable y reproducible) y usa un **LLM** únicamente para **redactar la justificación** de cada recomendación. No
reemplaza al reclutador: es una **herramienta de apoyo** para la preselección por mérito técnico.

> **Vacante de entrada:** *Machine Learning Systems Engineer, Research Tools* — **Anthropic**
> (equipo de *Encodings and Tokenization*). Requiere Python, ML systems / infraestructura, *data pipelines*,
> **tokenización (BPE, WordPiece)**, *encodings*, *performance optimization* y *distributed systems*.

> **Sobre el idioma:** los **CVs están en inglés** (igual que la vacante de Anthropic) para un emparejamiento
> candidato↔vacante monolingüe y más realista; el **análisis y los reportes del notebook están en español**
> (es el trabajo del curso).

---

## 2. Arquitectura

<img width="1691" height="3434" alt="image" src="https://github.com/user-attachments/assets/b4610db7-3805-4c24-bc0b-bbfdfd85a4f5" />

### 2.1 Pipeline en 6 etapas

| # | Etapa | Qué hace | Tecnología |
|---|-------|----------|------------|
| 1 | **Ingesta de la vacante** | Lee la descripción oficial desde el PDF y la estructura en campos (puesto, responsabilidades, requisitos, *skills* clave). Incluye un respaldo embebido si el PDF no está presente. | `pypdf` |
| 2 | **Base de conocimiento multimodal** | Materializa los 20 CVs en **16 PDF** y **4 PNG**, y luego **recupera su texto desde los archivos**: los PDF con extracción directa y los PNG con **OCR**. La base de conocimiento se construye *a partir de los archivos*, no de la estructura en memoria. | `fpdf2` (PDF), `Pillow` (PNG), `pytesseract` (OCR) |
| 3 | **Extracción estructurada** | Parsea de cada documento recuperado los campos clave (nivel, perfil, años, escolaridad, certificaciones, habilidades) para construir una **"base de datos de talentos"** que además enriquece los metadatos de la base vectorial. | `re` (regex) |
| 4 | **Vectorización + indexado** | Genera *embeddings* del texto de cada CV y los indexa en la base vectorial con distancia coseno. | `sentence-transformers` (`multilingual-e5-small`) + **ChromaDB** |
| 5 | **Búsqueda semántica** | Construye una consulta a partir de la vacante, la vectoriza y recupera los CVs más similares (top-*k* por similitud coseno). | ChromaDB |
| 6 | **Ranking híbrido + justificación** | Combina similitud semántica, *match* de *skills* y nivel para obtener el **Top-5**; un LLM redacta la **justificación** de cada candidato. | scoring propio + **Qwen2.5-1.5B-Instruct** |

### 2.2 Decisiones de diseño (y por qué)

- **Embeddings `multilingual-e5-small`.** Modelo **multilingüe** (soporta el contexto del curso) y **ligero** (corre en
  CPU). Se respetan los prefijos que el modelo recomienda: `query:` para la vacante y `passage:` para los CVs, lo que
  mejora la calidad de la recuperación.
- **ChromaDB con distancia coseno (persistente).** Base vectorial simple, reproducible y suficiente para 20 documentos;
  permite *re-runs* sin duplicar la colección.
- **Ranking híbrido, no solo similitud.** La similitud semántica por sí sola puede premiar CVs verbosos o no distinguir
  el nivel de *seniority*. Se combina con señales **estructuradas**:

  ```
  score = 0.60 · similitud_semántica  +  0.30 · match_skills  +  0.10 · ajuste_nivel
  ```

  donde `match_skills` es la cobertura de *skills* clave de la vacante presentes en el CV, y
  `ajuste_nivel` ∈ {Senior = 1.0, Semi-Senior = 0.7, Junior = 0.4} (el puesto pide perfil *experienced*). El resultado
  es un Top-5 **coherente y explicable** (cada *score* se descompone en sus tres componentes).
- **El LLM no decide el orden.** El ranking se calcula de forma **transparente y determinista**; el LLM solo **redacta**
  la justificación de cada candidato, que es donde aporta valor. Esto evita resultados irreproducibles.
- **Portabilidad y robustez.** El notebook **detecta el entorno** y se adapta: en **Colab + GPU** carga el LLM en
  **4-bit** (`bitsandbytes`); en **local + CPU** lo carga en **`bfloat16`** (~3 GB). Cada etapa tiene *fallback*: si no
  hay `tesseract`, los PNG usan un respaldo de texto; si el LLM no puede cargarse, las justificaciones se generan por
  reglas. Así el cuaderno **se ejecuta de principio a fin en ambos entornos**.

### 2.3 Stack tecnológico

| Componente | Tecnología |
|---|---|
| *Embeddings* | `intfloat/multilingual-e5-small` |
| Base de datos vectorial | **ChromaDB** (coseno, persistente) |
| LLM (justificación) | `Qwen/Qwen2.5-1.5B-Instruct` (4-bit en GPU / `bfloat16` en CPU) |
| Generación / lectura PDF | `fpdf2` / `pypdf` |
| Generación / lectura PNG | `Pillow` / `pytesseract` (OCR) |
| Datos y utilidades | `pandas`, `numpy`, `matplotlib` |

---

## 3. Datos: 20 CVs sintéticos, anonimizados

Por privacidad **no se usan CVs reales**: se generan **20 CVs sintéticos con un LLM** (prompt documentado en el
notebook). Cada CV incluye los 8 campos solicitados (resumen, formación, experiencia, habilidades técnicas y blandas,
certificaciones, idiomas y proyectos) y está **anonimizado** (solo un ID `CAND-001…CAND-020`; sin nombre, foto, género,
edad, nacionalidad ni estado civil).

- **Niveles:** 5 Junior · 10 Semi-Senior · 5 Senior.
- **6 tipos de perfil**, mezclando **cercanos** a la vacante (ML Systems Engineer, Data Engineer, ML Scientist/Engineer)
  y **lejanos** (BI Analyst, Cloud/DevOps Engineer, Software Developer/Architect) — esto permite **validar** que el
  sistema sabe distinguir lo relevante de lo irrelevante.
- **Multimodal:** 16 PDF + **4 PNG** (1 Junior, 2 Semi-Senior, 1 Senior).

Los 20 archivos generados están en [`CVs_generados/`](CVs_generados).

---

## 4. Cómo ejecutarlo

El notebook **detecta automáticamente el entorno** (Google Colab con GPU vs. local con CPU) y se adapta.

### Opción A — Google Colab (recomendado)
1. Sube el `.ipynb` y el PDF de la vacante a Colab.
2. *Entorno de ejecución → Ejecutar todo.* La primera celda instala dependencias y `tesseract` (OCR);
   el LLM se carga en 4-bit sobre GPU.

### Opción B — Local (CPU)
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# OCR opcional (si no está, los PNG usan un respaldo de texto):
#   Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-spa
#   macOS (brew):  brew install tesseract
jupyter notebook   # abre y ejecuta el notebook
```
En CPU el LLM se carga en `bfloat16` (~3 GB). La primera ejecución descarga los modelos de Hugging Face
(`intfloat/multilingual-e5-small` ≈ 470 MB y `Qwen/Qwen2.5-1.5B-Instruct` ≈ 3 GB).

---

## 5. Resultados (ejecución de referencia en CPU)

| # | Candidato | Perfil (profile) | Nivel | Score | Similitud | Skills |
|---|-----------|------------------|-------|-------|-----------|--------|
| 1 | CAND-016 | ML Systems Engineer | Senior | **0.901** | 0.946 | 14 |
| 2 | CAND-006 | ML Systems Engineer | Semi-Senior | 0.813 | 0.932 | 11 |
| 3 | CAND-007 | ML Systems Engineer | Semi-Senior | 0.757 | 0.922 | 8 |
| 4 | CAND-018 | ML Scientist/Engineer | Senior | 0.731 | 0.912 | 5 |
| 5 | CAND-001 | ML Systems Engineer | Junior | 0.724 | 0.919 | 8 |

- **Precision@5 = 1.00**: los 5 candidatos pertenecen a perfiles relevantes para la vacante.
- **Separación clara:** cobertura media de *skills* de **0.51** en el Top-5 frente a **0.13** en el resto del repositorio.
- **El ranking híbrido aporta valor:** el **Junior** CAND-001 obtuvo una similitud (0.919) **superior** a la del
  **Senior** CAND-018 (0.912) — los *embeddings* no distinguen *seniority* —, pero el **ajuste por nivel** coloca al
  Senior por encima (0.731 vs 0.724) y al Junior al final del Top-5, que es lo razonable para un puesto *experienced*.
- **Multimodal validado:** los 4 CVs en PNG (recuperados por OCR) participan en la búsqueda igual que los PDF.

Análisis completo, evaluación como experto del área y propuestas de métricas adicionales en
[`REPORTE_TECNICO.md`](REPORTE_TECNICO.md) y en las secciones 9–11 del notebook.

---

## 6. Estructura del repositorio

```
.
├── Gerardo Gonzalez Martinez_BusquedaDeTalento.ipynb   # Entregable principal (ejecutado, con salidas)
├── REPORTE_TECNICO.md                                  # Arquitectura, comportamiento y evaluación experta
├── README.md
├── requirements.txt
├── CVs_generados/                                       # 20 CVs sintéticos generados (16 PDF + 4 PNG, en inglés)
├── diagramas/
│   ├── arquitectura.excalidraw                          # Diagrama editable
│   ├── arquitectura.svg                                 # Render para el README/GitHub
│   └── DiagramNotebook.png                              # Diagrama embebido en la sección 6 del notebook
├── Job Application ... Anthropic.pdf                    # Vacante real (entrada del sistema)
└── MNA_NLP_Actividad_Busqueda_de_Talento_Inteligente.pdf   # Instrucciones de la actividad
```

---

## 7. Ética y anonimización

Los CVs **no incluyen** nombre, fotografía, género, edad, nacionalidad ni estado civil. Omitir estos atributos
**reduce el sesgo** (no predicen el desempeño y son fuente de discriminación), favorece el **cumplimiento legal**
(no discriminación en el empleo) y aplica **privacidad por diseño** (se recoge solo lo necesario para evaluar
idoneidad técnica). El sistema **no sustituye** al reclutador: prioriza por mérito técnico y la decisión final es
humana.
