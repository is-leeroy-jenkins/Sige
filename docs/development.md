# Development

## 🧭 Purpose

This page documents the validation workflow for maintaining Schedule-X documentation and source-generated API pages with MkDocs.

## 🧱 Source Layout

Schedule-X is currently centered on a single Streamlit application module:

```text
app.py
```

Documentation files are stored under:

```text
docs/
```

## 🧪 Local Validation

Run these commands before publishing documentation:

```powershell
python -m py_compile .\app.py
python -m compileall .
mkdocs build
```

## 📚 Documentation Build

The MkDocs site should build without missing-nav warnings, unlisted-page warnings, import failures, or avoidable griffe warnings.

## ✅ Docstring Rules

| Rule                | Requirement                                                                                     |
|---------------------|-------------------------------------------------------------------------------------------------|
| Style               | Use Google-style docstrings.                                                                    |
| Sections            | Use `Purpose:`, `Args:`, `Returns:`, `Raises:`, `Notes:`, and `Examples:` only when applicable. |
| Args                | Every documented argument must match the function signature.                                    |
| Returns             | Return descriptions must include an explicit type when the function lacks a return annotation.  |
| Procedures          | Do not add `Returns:` sections to procedures that do not return a meaningful value.             |
| Source preservation | Do not change runtime behavior while improving documentation.                                   |

## 🛠️ Common MkDocs Issues

| Warning or Error                | Cause                                                     | Fix                                                                                       |
|---------------------------------|-----------------------------------------------------------|-------------------------------------------------------------------------------------------|
| Page exists but is not in `nav` | Markdown file is not listed in `mkdocs.yml`.              | Add the page to `nav` or remove the unused file.                                          |
| Nav entry not found             | `mkdocs.yml` references a missing file.                   | Create the file or remove the nav entry.                                                  |
| mkdocstrings import error       | The module cannot be imported during documentation build. | Run direct Python import checks and fix missing dependencies or top-level runtime errors. |
| griffe argument warning         | Docstring argument entry does not match the signature.    | Correct the `Args:` section.                                                              |
| griffe return warning           | Return type cannot be inferred.                           | Add a return annotation or explicit typed `Returns:` entry.                               |

## 🚀 GitHub Pages Workflow

After the local build is clean, publish with the project’s selected GitHub Pages approach. A common MkDocs deployment command is:

```powershell
mkdocs gh-deploy --force
```

Run deployment only after `mkdocs build` succeeds locally.
