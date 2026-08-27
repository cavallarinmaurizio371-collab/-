from pathlib import Path
import sys

PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT))

from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas

from src.safety.path_guard import assert_safe_path


def main():
    output=assert_safe_path(PROJECT_ROOT/"output"/"pdf"/"calibration_chessboard_9x6_25mm.pdf")
    output.parent.mkdir(parents=True,exist_ok=True)
    page_width,page_height=landscape(A4)
    square=25*mm; columns,rows=10,7  # 9x6 inner corners.
    board_width,board_height=columns*square,rows*square
    origin_x=(page_width-board_width)/2; origin_y=(page_height-board_height)/2
    pdf=Canvas(str(output),pagesize=(page_width,page_height),pageCompression=0)
    pdf.setFillColorRGB(1,1,1); pdf.rect(0,0,page_width,page_height,fill=1,stroke=0)
    pdf.setFillColorRGB(0,0,0)
    for row in range(rows):
        for column in range(columns):
            if (row+column)%2==0:
                pdf.rect(origin_x+column*square,origin_y+row*square,square,square,fill=1,stroke=0)
    pdf.showPage(); pdf.save()
    print(output)


if __name__=="__main__": main()
