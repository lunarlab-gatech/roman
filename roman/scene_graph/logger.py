# log_config.py
import logging
from rich.logging import RichHandler
import rerun as rr

class MyFilter(logging.Filter):
    def filter(self, record):
        return record.name.startswith("Scene Graph 3D Logger")
    
handler = RichHandler(markup=True, rich_tracebacks=True)
handler.addFilter(MyFilter())

rr_handler = rr.LoggingHandler("/logs")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        handler,
        rr_handler
    ]
)

logger = logging.getLogger("Scene Graph 3D Logger")
