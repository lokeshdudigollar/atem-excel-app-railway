from flask import Flask, render_template, request, send_file, session
from PIL import Image
import pytesseract
import pandas as pd
import os
import cv2
import numpy as np
import re
from io import BytesIO

app = Flask(__name__)
app.secret_key = 'ikolr'
UPLOAD_FOLDER = '/tmp/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Expected fields
expected_fields = [
    "Full Name", "Gender", "DOB", "SSN", "Address 1", "Address 2",
    "City", "State", "Postal", "Country", "Email", "Contact",

    "Customer ID", "A/c Type", "A/c Name", "A/c Number", "IBAN", "BIC",
    "BTC Address", "ETH Address", "LTC Address", "CC_No", "Last Txn Amount","Last Txn Date", 
    
    "Company", "BS","EIN", "SkKLL Description" "ISIN", "Coupon",
    "Invested Amount", "Maturity Date", "Bond Name", "Bond Class",

    "Department", "Eanl3", "Product Name", "Unit Price", "User", "Purchase Token","Buying IPvé", "Buying IPv6é", 
    
    "Type", "Model", "Manufacturer", "VIN",

    "Beneficiary", "INS No.", 

    "Advisor ID", "Name","Contact", "Address",

    "Manager ID", "Name","Contact", "Address",

    "Manager ID", "Name","Contact", "Address"
]

extracted_data = []

def preprocess_image(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    img = cv2.fastNlMeansDenoising(img, h=30)
    temp_path = filepath.replace('.jpg', '_processed.jpg')
    cv2.imwrite(temp_path, img)
    return temp_path

def is_zero_based_on_inner_dot(char_img):
    gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    inner_contours = 0
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i in range(len(hierarchy)):
            if hierarchy[i][3] != -1:
                inner_contours += 1
    return inner_contours >= 1

@app.route('/', methods=['GET', 'POST'])
def index():
    global extracted_data
    if request.method == 'POST':
        if 'image' in request.files:
            file = request.files['image']
            if file:
                print("image got...", file)
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                print("filepath: ",filepath) #/tmp/uploads/18.jpg
                download_file_path = filepath.replace("/tmp/uploads/", "")
                file.save(filepath)
                filepath = preprocess_image(filepath)
                session['image_filepath'] = filepath

                image = cv2.imread(filepath)
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT, config="--psm 6")

                texts = data.get('text')
                confs = data.get('conf')
                xs = data.get('left')
                ys = data.get('top')
                ws = data.get('width')
                hs = data.get('height')

                cleaned_text = []
                cleaned_conf = []

                for t, c, x, y, w, h in zip(texts, confs, xs, ys, ws, hs):
                    if t.strip():
                        if t in ['0', 'O']:
                            crop = image[y:y+h, x:x+w]
                            t = '0' if is_zero_based_on_inner_dot(crop) else 'O'
                        cleaned_text.append(t)
                        cleaned_conf.append(int(c))

                extracted_data = split_into_key_value(cleaned_text, cleaned_conf, expected_fields)
                image_name = re.search(r'[/\\](\d+)_processed\.jpg$', filepath) #uploads\1222_processed.jpg
                print("image_name:",image_name)
                return render_template('edit.html', data=extracted_data, image_name=download_file_path)

        else:
            return 'No image file uploaded', 400
    return render_template('index.html')

#correction for IBAN
def correct_iban(iban):
    correction_map = {
        'O': '0', 
        'D': '0', 
        'Q': '0',
        'I': '1', 
        'L': '1', 
        'l': '1',
        'Z': '2',
        'S': '5',
        'G': '8', 
        'B': '8',
        'T': '7',
        'g': '9',
    }
    iban = iban.replace(' ', '')  # Remove spaces

    if len(iban) < 4:
        return iban  # Too short to fix

    first_two = iban[:2]
    next_two = list(iban[2:4])
    last_8 = list(iban[8:])   

    for i in range(2):
        if not next_two[i].isdigit():
            next_two[i] = correction_map.get(next_two[i].upper(), '0')  # fallback to '0' if unknown

    for i in range(8):
        if not last_8[i].isdigit():
            last_8[i] = correction_map.get(next_two[i], '0')

    corrected_iban = first_two + ''.join(next_two) + "****" + ''.join(last_8) #iban[8:]
    return corrected_iban



def correct_bic(bic):
    correction_map_digit = {
        'O': '0', 
        'D': '0',
        'I': '1', 
        'L': '1', 
        'l': '1',
        'Z': '2',
        'S': '5',
        'B': '8',
        'g': '9',
    }
    bic = bic.replace(' ', '')  # Remove spaces

    if len(bic) < 8:
        return bic  # Too short, just return

    corrected_bic = ''

    for idx, char in enumerate(bic):
        if idx <= 6:
            # First 7 must be letters
            corrected_bic += char.upper()  # Force to uppercase letter
        else:
            # After that must be digits
            if char.isdigit():
                corrected_bic += char
            else:
                corrected_bic += correction_map_digit.get(char.upper(), '0')  # fallback to 0

    return corrected_bic

def correct_btc_address(btc_address):
    correction_map = {
        'O': '0', 
        'D': '0',
        'I': '1', 
        'L': '1', 
        'l': '1',
        'Z': '2',
        'S': '5',
        'B': '8',
        'g': '9',
    }
    
    btc_address = btc_address.replace(' ', '')  # Remove spaces

    corrected_address = ''
    for char in btc_address:
        if char in correction_map:
            corrected_address += correction_map[char]
        elif char.upper() in correction_map:
            corrected_address += correction_map[char.upper()]
        else:
            corrected_address += char

    return corrected_address


def fix_day_in_date(text):
    # Correction map
    ocr_correction = {
        'L': '1',
        'I': '1',
        'O': '0',
        'Z': '2',
        'S': '5',
        'B': '8'
    }
    tokens = text.split()
    if tokens:
        day_part = tokens[0]  # e.g., 'LOth'
        fixed_day = ""
        for char in day_part[:2]:  # Only first two chars
            if char.isdigit():
                fixed_day += char
            elif char in ocr_correction:
                fixed_day += ocr_correction[char]
            else:
                fixed_day += char  # leave as-is (might still be wrong)
        fixed_day += day_part[2:]  # add 'th' etc.

        tokens[0] = fixed_day  # replace corrected day back
        return ' '.join(tokens)
    return text

def fix_advisorId(value: str) -> str:
    # Step 1: Clean spaces
    value = value.replace(' ', '')

    if len(value) != 10:
        # Optional: handle differently if wrong length
        pass

    first6 = list(value[:6])
    last4 = list(value[6:])

    # Step 2: Fix first 5 letters
    for i in range(6):
        if first6[i].isdigit():
            if first6[i] == '0':
                first6[i] = 'O'
            elif first6[i] == '1':
                first6[i] = 'I'
            elif first6[i] == '5':
                first6[i] = 'S'
            else:
                first6[i] = chr(65 + int(first6[i]) % 26)  # fallback to random letter

    # Step 3: Fix last 5 digits
    for i in range(4):
        if last4[i].isalpha():
            if last4[i].upper() == 'S':
                last4[i] = '5'
            elif last4[i].upper() in ['I', 'L']:
                last4[i] = '1'
            elif last4[i].upper() == 'O':
                last4[i] = '0'
            else:
                last4[i] = '0'  # fallback to '0'

    # Step 4: Apply special Q0 logic
    # if last4[0] == '0' and first6[4].upper() == 'Q':
    #     corrected_value = ''.join(first6[:4]) + 'Q' + ''.join(last4[1:])
    # else:
    #     corrected_value = ''.join(first6) + ''.join(last4)
    corrected_value = ''.join(first6) + ''.join(last4)
    # Step 5: Final safety
    corrected_value = corrected_value[:10]

    return corrected_value

def fix_common_digit_errors2(value: str) -> str:
    correction_map = {
        'G': '9',
        'g': '9',
        'O': '0',
        'I': '1',
        'l': '1',
        'Z': '2',
        'S': '5',
        'B': '8',
        'A': '4',  # sometimes OCR confuses 4 as A
        '2': '4'   # only in specific context, maybe restrict this
    }

    corrected = []
    for char in value:
        if char in correction_map:
            corrected.append(correction_map[char])
        else:
            corrected.append(char)
    return ''.join(corrected)

def fix_common_digit_errors(value: str) -> str:
    # Replace misread '2' with '4' if certain conditions met
    corrected = list(value)

    for i, ch in enumerate(corrected):
        if ch == '2':
            # Look left and right neighbors if available
            left = corrected[i-1] if i-1 >= 0 else ''
            right = corrected[i+1] if i+1 < len(corrected) else ''

            # Heuristic 1: if surrounded by 4s or 8s, more likely misread
            if left in ['4'] or right in ['4']:
                corrected[i] = '4'
                
            # Heuristic 2: if position is middle digit of 3-3-4 pattern (for phone numbers)
            if len(value.replace('.', '').replace('-', '')) == 10:  # probably phone number
                if i in [4, 5]:  # typical area
                    corrected[i] = '4'

    return ''.join(corrected)

def fix_common_letter_errors(value: str) -> str:
    corrected = list(value)

    for i, ch in enumerate(corrected):
        # Heuristic: if field expects letters, fix common misreads
        if ch == 'I':
            # If surrounded by letters or expected letter positions, prefer J
            left = corrected[i-1] if i-1 >= 0 else ''
            right = corrected[i+1] if i+1 < len(corrected) else ''
            print(left, right, flush=True)

            if (left.isalpha() or right.isalpha()):
                corrected[i] = 'J'

    return ''.join(corrected)

def fix_beneficiary_id(value: str) -> str:
    pcorrection_map = {
        'O': '0', 
        'D': '0', 
        'Q': '0',
        'I': '1', 
        'L': '1', 
        'l': '1',
        'Z': '2',
        'S': '5',
        'G': '8', 
        'B': '8',
        'T': '7',
        'g': '9',
    }
    value = value.replace(' ', '')  # Remove spaces
    expected_length = 13

    if len(value) == expected_length:
        return value

    # If too long, try removing characters one by one
    if len(value) > expected_length:
        index_of_7 = value.find('7')
        if index_of_7 != -1:
            value = value.replace(value[index_of_7+1], '')

    return value


def split_into_key_value(cleaned_text, cleaned_conf, expected_fields):
    advisor_id_corrected = False # for Advisor ID correction to correct first advisor ID
    contact_first = False # for Contact correction to correct first contact number
    results = []
    n = len(cleaned_text)
    words_with_conf = [(cleaned_text[i], cleaned_conf[i]) for i in range(n)]

    i = 0
    corrected = False
    wronged = False
    yellowed = False
    suggestions = []
    ac_name_field = False #to fix a/c name field text
    while i < n:
        match_found = False
        for field in expected_fields:
            field_parts = field.split()
            field_len = len(field_parts)

            if i + field_len <= n and all(words_with_conf[i + j][0] == field_parts[j] for j in range(field_len)):
                match_found = True
                i += field_len

                value_words = []
                value_confs = []

                while i < n:
                    is_next_field = False
                    for next_field in expected_fields:
                        next_parts = next_field.split()
                        next_len = len(next_parts)
                        if i + next_len <= n and all(words_with_conf[i + k][0] == next_parts[k] for k in range(next_len)):
                            is_next_field = True
                            break
                    if is_next_field:
                        break
                    value_words.append(words_with_conf[i][0])
                    value_confs.append(words_with_conf[i][1])
                    i += 1

                value = ' '.join(value_words).strip()

                #colors
                #reds
                if field in ['BTC Address', 'ETH Address', 'LTC Address', 'Purchase Token', 'Manager ID', 'Buying IPv6','Buying IPv6é']:
                    wronged = True
                    corrected = False
                
                #no color
                elif field in ['DOB', 'State', 'Postal', 'Email', 'CC_No', 'Address 1', 'Address 2']:
                    corrected = False
                    wronged = False
                #green
                elif field in ['Company', 'BS', 'EIN', 'Skll Description', 'IBAN', 'BIC', 'ISIN','Advisor ID']:
                    corrected = True
                    wronged = False
                #yellow
                elif field in ['Customer ID', 'Beneficiary', 'A/c Number']:
                    yellowed = True
                else:
                    wronged = False
                    corrected = False
                    yellowed = False
                # Apply post-correction for specific fields
                if field in ['DOB', 'Maturity Date', 'Address']:
                    value = value.replace('@', '0')

                if field == 'Contact':
                    if not contact_first:
                        value = "+1 (xxx) xxx " + value[-4:]
                        contact_first = True  # mark it corrected
                # Fix future year typos in Maturity Date
                if field == 'Maturity Date':
                    match = re.match(r'(\d{2})[ -]([A-Z]{3})[ -](\d{4})', value)
                    if match:
                        day, month, year = match.groups()
                        year_int = int(year)
                        if year_int > 2025:
                            corrected_year = year[:1] + '0' + year[2:]
                            corrected_year_int = int(corrected_year)
                            # Only apply if corrected year is realistic (like 1900–2025)
                            if 1900 <= corrected_year_int <= 2025:
                                value = f"{day}-{month}-{corrected_year}"
                
                #Fix for Company field
                if field == 'Company':
                    tokens = value.split()
                    # Normalize tokens for matching keywords
                    lowered_tokens = [t.lower() for t in tokens]

                    try:
                        bs_index = tokens.index('BS')
                        ein_index = tokens.index('EIN')
                        skll_index = None
                        for j in range(len(lowered_tokens) - 1):
                            if lowered_tokens[j] == 'skll' and lowered_tokens[j + 1] == 'description':
                                skll_index = j
                                break

                        if bs_index < ein_index < skll_index:
                            company_val = ' '.join(tokens[:bs_index]).strip()
                            bs_val = ' '.join(tokens[bs_index + 1:ein_index]).strip()
                            ein_val = ' '.join(tokens[ein_index + 1:skll_index]).strip()
                            skll_val = ' '.join(tokens[skll_index + 2:]).strip()

                            confidence = round(sum(value_confs) / len(value_confs), 2) if value_confs else 0

                            results.append({"field": "Company", "value": company_val, "confidence": confidence, "corrected": True, "wronged": False})
                            results.append({"field": "BS", "value": bs_val, "confidence": confidence,"corrected": True,"wronged": False})
                            results.append({"field": "EIN", "value": ein_val, "confidence": confidence, "corrected": True,"wronged": False})
                            results.append({"field": "Skll Description", "value": skll_val, "confidence": confidence, "corrected": True,"wronged": False})
                        else:
                            # fallback to whole value
                            results.append({"field": "Company", "value": value, "confidence": confidence})

                    except Exception as e:
                        results.append({"field": "Company", "value": value, "confidence": confidence})
                    break;
                if field == "EIN" and len(value) > 13:
                    fullValue = value
                    value = value[:14]
                    value = value.replace(" ", "")

                    isin_index = fullValue.index('ISIN')
                    skll = fullValue[14:isin_index]
                    skll_description = skll[skll.index("Description")+len("Description"):]
                    skll_description_spaces_removed = skll_description.strip()

                    isin_value = fullValue[isin_index+5:]
                    isin_value_updated = isin_value.replace("@","0")
                    isin_value_updated = fix_common_letter_errors(isin_value_updated)
                    results.append({"field": "Skll Description", "value": skll_description_spaces_removed, "confidence": confidence, "corrected": True,"wronged": False})
                    results.append({"field": "ISIN", "value": isin_value_updated, "confidence": confidence, "corrected": True,"wronged": False})

                if field == "Skll Description":
                    value = value.replace("SkKLL Description", "")
                    value = value.replace(" Skll Description ", "")

                    
                
                if field == 'Bond Class':
                    value = value.replace('Last Purchase Detail', '')
                if field == 'INS No.':
                    value = value.replace('Account Advisor', '')
                if field == 'Address':
                    value = value.replace('Investment Advisor', '')
                    value = value.replace('Insurance Manager', '')
                    value = value.replace('Assets Manager', '')
                if field == 'VIN':
                    value = value.replace('Insurance Detail', '')
                if field == 'Beneficiary':
                    field = 'Beneficiary Identifier ID'
                    value = value.replace('Identifier ID', '')
                    value = fix_beneficiary_id(value)
                if field == 'A/c Number':
                    value = "****" + value[-6:]
                    if value[-6] == "B":
                        value = value[:-6] + "8" + value[-5:]
                    value = fix_common_digit_errors(value)
                if field == "Address":
                    value = fix_common_digit_errors(value)
                    if "444" in value:
                        value = value.replace("444", "44", 1)
                if field == 'IBAN':
                    value = correct_iban(value)

                if field == 'BTC Address':
                    value = correct_btc_address(value)

                if field == 'BIC':
                    value = correct_bic(value)
                
                if field == 'ETH Address':
                    value = '0x' + value[2:]
                if field == 'CC_No':
                    value = value.replace(' ', '')
                    value = '***-' + value[3:]
                if field == 'Last Txn Date':
                    value = fix_day_in_date(value)
                    value =  re.sub(r'(?<=\d{2}/[A-Z]{3}/)(1L7|IL7|I7|L7)', '17', value, flags=re.IGNORECASE)
                if field == 'Buying IPvé':
                    field = 'Buying IPv4'
                
                if field == 'Buying IPv6é':
                    field = 'Buying IPv6'
                    value = value.replace('Vehicle Detail', '')

                if field == 'Advisor ID':
                    if not advisor_id_corrected:
                        value = fix_advisorId(value)
                        advisor_id_corrected = True  # mark it corrected
                if field == 'Eanl3':
                    value = value.replace('Q', '0')

                if field  == "ISIN" or field == "Advisor ID" or field == "Manager ID":
                    value = fix_common_letter_errors(value)

                if field  == "ISIN":
                    value = value.replace("@", "0")

                if field == "A/c Type" and len(value) > len("invice"):
                    value = value.split()[0]

                if field == "Name" and not ac_name_field:
                    field = "A/c Name"
                    ac_name_field = True

                #remove extra spaces at start and end
                if field == "Address":
                    value = value.strip()
                if field == "VIN":
                    value = value.replace(" ", "")
                if field == "A/c Number":
                    value = fix_common_digit_errors2(value)

                #setting suggestions
                if field == "Customer ID" or field == "Customer":
                    suggestion = "May be it always starts with 1001?"
                elif field == "Manager ID" and len(value) > 0:
                    suggestion = "check for commom letters mistakes like H/M, J"
                elif field == "A/c Number" or field == "Number":
                    suggestion = "check for common mistakes like G/9 or 2/4"
                elif field == "IBAN":
                    suggestion = "check for 0s mistakes- 0/9, 9/0"
                elif field == "CC_No":
                    suggestion = "check for 0s mistakes- 0/9, 9/0"
                elif field == "Beneficiary" or field == "Beneficiary Identifier ID":
                    suggestion = "check for 5/S or extra T might be replaced around 7"
                else:
                    suggestion = ""
                confidence = round(sum(value_confs) / len(value_confs), 2) if value_confs else 0
                results.append({
                    "field": field, 
                    "value": value, 
                    "corrected": corrected, 
                    "wronged": wronged, 
                    "yellowed":yellowed,
                    "suggestion": suggestion
                })
                suggestion = ""
                break

        if not match_found:
            i += 1

    return results

@app.route('/edit', methods=['GET', 'POST'])
def edit():
    global extracted_data
    image_filepath = session.get('image_filepath')
    if not image_filepath:
        return "No file uploaded", 400

    if request.method == 'POST':
        updated_values = request.form.getlist('value')
        updated_data = []
        for idx, item in enumerate(extracted_data):
            field = item['field']
            #confidence = item['confidence']
            suggestion = item['suggestion']
            updated_value = updated_values[idx] if idx < len(updated_values) else item['value']
            updated_data.append({"Field": field, "Value": updated_value, "suggestion": suggestion})

        df = pd.DataFrame(updated_data)
        excel_path = os.path.join(UPLOAD_FOLDER, 'final_output.xlsx')
        df.to_excel(excel_path, index=False)

        return send_file(excel_path, as_attachment=True)

    return render_template('edit.html', data=extracted_data)


@app.route("/download_excel", methods=["POST"])
def download_excel():
    image_name = request.form.get("image_name", "corrected_data")
    total_rows = int(request.form["total_rows"])
    records = []
    for i in range(total_rows):
        field = request.form.get(f"field_{i}")
        value = request.form.get(f"value_{i}")
        records.append({"Field": field, "Value": value})

    df = pd.DataFrame(records)

    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name="Corrected Data", startrow=0, header=False)

        workbook = writer.book
        worksheet = writer.sheets["Corrected Data"]

        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#D9D9D9',
            'align': 'center',
            'valign': 'vcenter',
            'border': 1
        })

        subheader_format = workbook.add_format({
            'bold': True,
            'bg_color': '#f0f0f0',
            'align': 'left',
            'valign': 'vcenter',
            'border': 1
        })

        data_format = workbook.add_format({
            'border': 1
        })

        # Define section structure
        sections = [
            ("Personal Information", 0, 12),
            ("Account Information", 12, 12),
            ("Investment Information", 24, 10),
            ("Assets & Last Purchase Information", 34, 14),
            ("Legal Advisors", 48, len(df) - 48)
        ]

        current_row = 0
        for title, start, count in sections:
            # Main Section Header
            worksheet.merge_range(current_row, 0, current_row, 1, title, header_format)
            current_row += 1

            if title == "Assets & Last Purchase Information":
                # Subheader: Last Purchase Detail
                worksheet.merge_range(current_row, 0, current_row, 1, "Last Purchase Detail", subheader_format)
                current_row += 1
                for i in range(8):
                    worksheet.write(current_row, 0, df.iloc[start + i]["Field"], data_format)
                    worksheet.write(current_row, 1, df.iloc[start + i]["Value"], data_format)
                    current_row += 1

                # Subheader: Vehicle Detail
                worksheet.merge_range(current_row, 0, current_row, 1, "Vehicle Detail", subheader_format)
                current_row += 1
                for i in range(8, 12):
                    worksheet.write(current_row, 0, df.iloc[start + i]["Field"], data_format)
                    worksheet.write(current_row, 1, df.iloc[start + i]["Value"], data_format)
                    current_row += 1

                # Subheader: Insurance Detail
                worksheet.merge_range(current_row, 0, current_row, 1, "Insurance Detail", subheader_format)
                current_row += 1
                for i in range(12, 14):
                    worksheet.write(current_row, 0, df.iloc[start + i]["Field"], data_format)
                    worksheet.write(current_row, 1, df.iloc[start + i]["Value"], data_format)
                    current_row += 1
                
            elif title == "Legal Advisors":
                sub_sections = [
                    ("Account Advisor", 0, 4),
                    ("Assets Manager", 4, 4),
                    ("Investment Advisor", 8, 4),
                    ("Insurance Manager", 12, count - 12),  # Remaining
                ]
                for sub_title, offset, sub_count in sub_sections:
                    worksheet.merge_range(current_row, 0, current_row, 1, sub_title, subheader_format)
                    current_row += 1
                    for i in range(sub_count):
                        worksheet.write(current_row, 0, df.iloc[start + offset + i]["Field"], data_format)
                        worksheet.write(current_row, 1, df.iloc[start + offset + i]["Value"], data_format)
                        current_row += 1
            
            else:
                for i in range(count):
                    worksheet.write(current_row, 0, df.iloc[start + i]["Field"], data_format)
                    worksheet.write(current_row, 1, df.iloc[start + i]["Value"], data_format)
                    current_row += 1

        # Set column widths
        worksheet.set_column("A:A", 30)
        worksheet.set_column("B:B", 40)

    output.seek(0)
    return send_file(
        output,
        download_name=f"{image_name.rsplit('.', 1)[0]}.xlsx",
        as_attachment=True,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")