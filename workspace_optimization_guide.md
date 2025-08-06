# Workspace Optimization Guide

## Large Files Found
The following files are larger than 1MB and may cause Claude API timeouts:

- .\assets\images\unity_econometric_analysis.png
- .\een\Lib\site-packages\30fcd23745efe32ce681__mypyc.cp313-win_amd64.pyd
- .\een\Lib\site-packages\altair\vegalite\v5\schema\channels.py
- .\een\Lib\site-packages\altair\vegalite\v5\schema\core.py
- .\een\Lib\site-packages\altair\vegalite\v5\schema\vega-lite-schema.json
- .\een\Lib\site-packages\cryptography\hazmat\bindings\_rust.pyd
- .\een\Lib\site-packages\dash\dash-renderer\build\dash_renderer.dev.js
- .\een\Lib\site-packages\dash\dash_table\async-export.js.map
- .\een\Lib\site-packages\dash\dash_table\async-table.js.map
- .\een\Lib\site-packages\dash\dcc\async-mathjax.js

## Recommendations
1. Move large files to external storage
2. Compress large files where possible
3. Use .claudeignore to exclude unnecessary files
4. Break large operations into smaller chunks
5. Use file references instead of full content

## Quick Actions
1. Use the minimal .cursorrules file
2. Implement incremental development
3. Work on one file at a time
4. Use git for version control
5. Monitor workspace size regularly
