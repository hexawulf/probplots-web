"""Universal PNG plot renderer for probplots-web"""
import io
from fastapi import Response
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_to_png(fig, dpi=150):
    """Convert a matplotlib figure to PNG Response.
    
    Args:
        fig: matplotlib figure
        dpi: dots per inch for output
    
    Returns:
        FastAPI Response with image/png media type
    """
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")
