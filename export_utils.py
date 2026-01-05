from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib import colors

def export_to_pdf(output_path, transcript_data, speaker_names):
    """
    Generates a PDF from the transcript.
    transcript_data: list of {start, end, text, speaker}
    speaker_names: dict of {raw_speaker_label: display_name}
    """
    c = canvas.Canvas(output_path, pagesize=LETTER)
    width, height = LETTER
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(inch, height - 0.7*inch, "Video Transcript")
    c.line(inch, height - 0.75*inch, width - inch, height - 0.75*inch)
    
    y = height - 1.2 * inch
    
    for item in transcript_data:
        start_time = format_time(item['start'])
        raw_speaker = item.get('speaker', 'Unknown')
        display_name = speaker_names.get(raw_speaker, raw_speaker)
        text_content = item['text']
        
        # Check for page break
        if y < inch:
            c.showPage()
            y = height - inch
            c.setFont("Helvetica", 10)

        # Draw Timestamp
        c.setFont("Helvetica-Oblique", 8)
        c.setFillColor(colors.gray)
        c.drawString(inch, y, f"[{start_time}]")
        
        # Draw Speaker
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(colors.black)
        c.drawString(inch + 0.6*inch, y, f"{display_name}:")
        
        # Draw Text (Simple wrap logic for prototype)
        # For a robust app, use ReportLab Platypus Paragraphs, but this is simpler for now.
        c.setFont("Helvetica", 10)
        text_start_x = inch + 1.5*inch
        
        # Very basic wrapping by char count (not ideal but works for plain text)
        max_chars = 80
        words = text_content.split()
        current_line = ""
        
        # Initial line offset
        current_y = y
        
        for word in words:
            if len(current_line) + len(word) + 1 < max_chars:
                current_line += word + " "
            else:
                c.drawString(text_start_x, current_y, current_line)
                current_y -= 14
                current_line = word + " "
                
                # Check page break inside text block
                if current_y < inch:
                    c.showPage()
                    current_y = height - inch
                    c.setFont("Helvetica", 10)
        
        # Draw last line
        if current_line:
            c.drawString(text_start_x, current_y, current_line)
            
        # Move Y down for next block
        y = current_y - 20 

    c.save()

def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}" if h > 0 else f"{int(m):02d}:{int(s):02d}"
