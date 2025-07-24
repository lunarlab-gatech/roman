# log_config.py
import logging
from rich.logging import RichHandler


class MyFilter(logging.Filter):
    def filter(self, record):
        return record.name.startswith("Scene Graph 3D Logger")
    
handler = RichHandler(markup=True, rich_tracebacks=True)
handler.addFilter(MyFilter())

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        handler
    ]
)

logger = logging.getLogger("Scene Graph 3D Logger")
