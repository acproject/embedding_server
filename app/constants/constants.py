'''span 类型'''


CROSS_PAGE = 'cross_page' # 跨页
SAME_PAGE = 'same_page' # 同页
SAME_PAGE_AND_CROSS_PAGE = 'same_page_and_cross_page' # 同页和跨页
SAME_PAGE_AND_CROSS_PAGE_AND_SAME_PAGE = 'same_page_and_cross_page_and_same_page' # 同页和跨页和同页

'''block 类型'''
BLOCK = 'block' # 块
INLINE = 'inline' # 行内
INLINE_BLOCK = 'inline_block' # 行内块

PAESE_TYPE = {
    TEXT : 'text',          # 文本类型，用于可直接解析的
    IMAGE: 'image'          # 图片类型，用于OCR识别的
}

OCR_CONTENT_TYPE = {
    IMAGE: 'image',
    TABLE : 'table',
    TEXT : 'text',
    INLINE_EQUATION : 'inline_equation',
}

OCR_BLOCK_TYPE = {
    Image : 'image',
    ImageBody : 'image_body',
    ImageCaption : 'image_caption',
    ImageFigure : 'image_figure',
    ImageFooter: 'image_footer',
    ImageHeader : 'image_header',
    ImageParagraph : 'image_paragraph',
    ImageSpan : 'image_span',
    ImageTable : 'image_table',
    ImageTableCell : 'image_table_cell',
    ImageTableRow : 'image_table_row',
    ImageTableBody : 'image_table_body',
    ImageTableHeader : 'image_table_header',
    ImageTableFooter : 'image_table_footer',
    Table:  'table',
    TableBody : 'table_body',
    TableCell : 'table_cell',
    TableFooter : 'table_footer',
    TableHeader : 'table_header',
    TableRow : 'table_row',
    Text : 'text',
    Title : 'title',
    Paragraph : 'paragraph',
    Span : 'span',
    Equation : 'equation',
    InlineEquation : 'inline_equation',
    Footnote : 'footnote',
    Caption : 'caption',
    Figure : 'figure',
    Header : 'header',
    Footer : 'footer',
    List : 'list',
    ListItem : 'list_item',
    Code : 'code',
    Quote : 'quote',
    HorizontalLine : 'horizontal_line',
    Equation : 'equation',
    InlineEquation : 'inline_equation',
}


OCR_CATEGORY = {
    Tilte : 0,
    Paragraph : 1,
    Image : 2,
    ImageBody : 21,
    ImageCaption : 22,
    ImageFigure : 23,
    ImageFooter: 24,
    ImageHeader : 25,
    ImageParagraph : 26,
    ImageSpan : 27,
    ImageTable : 28,
    ImageTableCell : 29,
    ImageTableRow : 30,
    ImageTableBody : 31,
    ImageTableHeader : 32,
    ImageTableFooter : 33,
    TableBody : 34,
    TableCell : 35,
    TableFooter : 36,
    TableHeader : 37,
    TableRow : 38,
    Table : 3,
    Equation : 4,
    InlineEquation : 5,
    Text : 6,
    Caption : 7,
    Figure : 8,
    Header : 9,
    Footer : 10,
    List : 11,
    ListItem : 12,
    Code : 13,
    Quote : 14,
    HorizontalLine : 15,
    OCRText : 16,
}