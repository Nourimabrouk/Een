#!/usr/bin/env python3
"""
Streamlit Cloud entrypoint: runs the consolidated Metastation HUD dashboard.
"""
from metastation_streamlit import main

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Streamlit Cloud entrypoint for the Een Unity Metastation dashboard.
Delegates to the consolidated HUD at `metastation_streamlit.py`.
"""

from metastation_streamlit import main

if __name__ == "__main__":
    main()
