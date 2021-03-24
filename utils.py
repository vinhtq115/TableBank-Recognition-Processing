from lxml import etree


def advanced_table_check(content):
    """
    Check if table annotation is advanced (mostly spanning row/column or error in annotation).
    :param content: Content of XML annotation file.
    :return: True if table annotation is advanced, else False.
    """
    if content.count('<tabular>') > 1 or content.count('<td>') > 0:
        return True
    return False


def get_file_name(file_path):
    """
    Return filename given path to file.
    :param file_path: Path to file.
    :return: Name of file without extension.
    """
    ff = file_path.split('\\')[-1]
    return ff[:ff.rfind('.')]


def count_cells(xml_file):
    """
    Return number of cells, including and excluding empty cells, rows and columns list.
    :param xml_file: Path to annotation XML file.
    :return: Number of cells, including and excluding empty cells, rows and columns list.
    """
    # Parse XML
    parser = etree.XMLParser(ns_clean=True, remove_blank_text=True)
    xml_tree = etree.parse(xml_file, parser)
    tabular = xml_tree.getroot()  # <tabular>

    # Read and store rows and columns
    rows = []  # Store rows and columns
    for i in tabular:  # Traverse <thead> and <tbody>
        for row in i:  # Traverse <tr>
            row_s = []
            for cell in row:
                if cell.tag == 'tdy':
                    row_s.append('tdy')
                elif cell.tag == 'tdn':
                    row_s.append('tdn')
            rows.append(row_s)

    total = 0  # Number of cells, including empty ones
    total_cell_with_data = 0  # Number of cells, excluding empty ones
    for idx, row in enumerate(rows, 1):
        total += len(row)
        for cell in row:
            if cell == 'tdy':
                total_cell_with_data += 1

    return total, total_cell_with_data, rows


def draw_border(img):
    h, w = img.shape
    for i in range(w):
        img[0][i] = 255
        img[h-1][i] = 255
    for i in range(h):
        img[i][0] = 255
        img[i][w-1] = 255