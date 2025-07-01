"""
Utilities, path handler
"""
from pathlib import Path
from typing import Dict, Optional, Union
import os
import pandas as pd


class PathManager:
    """
    Manages standard paths across the analysis modules.
    """
    
    BASE_DIR = Path(".")
    DATA_DIR = BASE_DIR / "data"
    CHARTS_DIR = BASE_DIR / "charts"
    INSIGHT_CHARTS_DIR = BASE_DIR / "insight_charts"
    DIRECT_CHARTS_DIR = BASE_DIR / "direct_charts"
    
    @classmethod
    def ensure_dirs(cls) -> None:
        """Create all standard directories if they don't exist."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.INSIGHT_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.DIRECT_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_data_path(cls, filename: str) -> Path:
        """Get path for a data file in the standard data directory."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        return cls.DATA_DIR / filename
    
    @classmethod
    def get_chart_path(cls, filename: str, chart_type: str = "general") -> Path:
        """
        Get path for a chart file in the appropriate charts directory.
        
        Args:
            filename: Name of the chart file
            chart_type: Type of chart (general, insight, direct)
            
        Returns:
            Path object for the chart file
        """
        if chart_type == "insight":
            output_dir = cls.INSIGHT_CHARTS_DIR
        elif chart_type == "direct":
            output_dir = cls.DIRECT_CHARTS_DIR
        else:
            output_dir = cls.CHARTS_DIR
            
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / filename


def save_dataframe(
    df: pd.DataFrame, 
    filename: str, 
    format_type: str = "excel", 
    sheet_name: str = "Sheet1", 
    index: bool = False
) -> Path:
    """
    Save a DataFrame to a file in the data directory.
    
    Args:
        df: DataFrame to save
        filename: Name for the output file (without extension)
        format_type: "excel" or "csv"
        sheet_name: Sheet name (for Excel only)
        index: Whether to include index in output
    
    Returns:
        Path to the saved file
    """
    if not filename.endswith((".xlsx", ".csv")):
        ext = ".xlsx" if format_type == "excel" else ".csv"
        filename = f"{filename}{ext}"
    
    file_path = PathManager.get_data_path(filename)
    
    if filename.endswith(".xlsx"):
        df.to_excel(file_path, sheet_name=sheet_name, index=index)
    else:
        df.to_csv(file_path, index=index)
    
    return file_path


def load_dataframe(filename: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    Load a DataFrame from a file in the data directory.
    
    Args:
        filename: Name of the file to load
        sheet_name: Sheet name (for Excel only)
    
    Returns:
        Loaded DataFrame
    """
    file_path = PathManager.get_data_path(filename)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.suffix == ".xlsx":
        return pd.read_excel(file_path, sheet_name=sheet_name)
    else:
        return pd.read_csv(file_path)


def save_figure(figure, filename: str, chart_type: str = "general", dpi: int = 300) -> Path:
    """
    Save a matplotlib figure to a file in the appropriate charts directory.
    
    Args:
        figure: Matplotlib figure to save
        filename: Name for the output file (without extension)
        chart_type: Type of chart (general, insight, direct)
        dpi: Resolution for saving the figure
    
    Returns:
        Path to the saved file
    """
    if not filename.endswith((".png", ".jpg", ".pdf")):
        filename = f"{filename}.pdf"
    
    file_path = PathManager.get_chart_path(filename, chart_type)
    figure.savefig(file_path, dpi=dpi, bbox_inches="tight")
    
    return file_path
