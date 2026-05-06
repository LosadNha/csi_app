"""Centralized colors, fonts, and palette for the application."""

from PyQt5.QtGui import QColor, QFont, QPalette


# ── Color tokens ──────────────────────────────────────────────────────────────

class Colors:
    # Backgrounds
    BG_WINDOW       = "#FAFAFA"
    BG_BASE         = "#FFFFFF"
    BG_ALT          = "#F5F5F5"
    BG_CARD         = "#F5F5F5"
    BG_DASHBOARD    = "#ECEFF1"

    # Borders / cards
    BORDER_NEUTRAL  = "#BDBDBD"
    BORDER_OK       = "#4CAF50"
    BORDER_WARN     = "#FFC107"
    BORDER_ERR      = "#F44336"

    CARD_OK         = "#E8F5E9"
    CARD_WARN       = "#FFF8E1"
    CARD_ERR        = "#FFEBEE"

    # Accent
    PRIMARY         = "#1976D2"
    PRIMARY_DARK    = "#1565C0"
    DANGER          = "#D32F2F"
    SUCCESS         = "#4CAF50"
    SIMULATE        = "#7B1FA2"

    # Text
    TEXT            = "#212121"
    TEXT_SECONDARY  = "#616161"
    TEXT_MUTED      = "#757575"

    # Plot colors
    PLOT_AMP        = "#1565C0"
    PLOT_PHASE      = "#C62828"

    # Chart palette (up to 10 devices)
    CHART_PALETTE = [
        "#1565C0", "#C62828", "#2E7D32", "#F57F17", "#6A1B9A",
        "#00838F", "#D84315", "#4E342E", "#37474F", "#AD1457",
    ]

    # Status badge
    STATUS_CONNECTED_BG    = SUCCESS
    STATUS_DISCONNECTED_BG = "#F44336"


# ── Font factory ──────────────────────────────────────────────────────────────

class Fonts:
    @staticmethod
    def mono(size: int = 8, bold: bool = False) -> QFont:
        f = QFont("Monospace", size, QFont.Bold if bold else QFont.Normal)
        f.setStyleHint(QFont.Monospace)
        return f

    @staticmethod
    def ui(size: int = 10, bold: bool = False) -> QFont:
        return QFont("Arial", size, QFont.Bold if bold else QFont.Normal)


# ── Application palette ───────────────────────────────────────────────────────

def make_palette() -> QPalette:
    p = QPalette()
    p.setColor(QPalette.Window,          QColor(Colors.BG_WINDOW))
    p.setColor(QPalette.WindowText,      QColor(Colors.TEXT))
    p.setColor(QPalette.Base,            QColor(Colors.BG_BASE))
    p.setColor(QPalette.AlternateBase,   QColor(Colors.BG_ALT))
    p.setColor(QPalette.ToolTipBase,     QColor(Colors.BG_BASE))
    p.setColor(QPalette.ToolTipText,     QColor(Colors.TEXT))
    p.setColor(QPalette.Text,            QColor(Colors.TEXT))
    p.setColor(QPalette.Button,          QColor("#E0E0E0"))
    p.setColor(QPalette.ButtonText,      QColor(Colors.TEXT))
    p.setColor(QPalette.Highlight,       QColor(Colors.PRIMARY))
    p.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    return p
