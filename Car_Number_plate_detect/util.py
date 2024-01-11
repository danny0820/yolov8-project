import string
import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)
O_to_nine=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] 
def license_complies_format(text):
    if len(text) >=5 and len(text)<=7:
        for i in text:
            if i not in string.ascii_uppercase and i  not in O_to_nine:
                return False
        return True
    return False


def format_license(text):
    license_plate_ = ''
    for j in range(len(text)):
        if text[j] =='o' or text[j] == "O":
            license_plate_ += '0'
        elif text[j] == "I":
            license_plate_ += '1'
        else:
            license_plate_ += text[j]
    return license_plate_


def read_license_plate(license_plate_crop):
  

    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection
        if score<0.6:
            continue
        text = text.upper().replace(' ', '')
        if license_complies_format(text):
            return format_license(text)
            
    return None

