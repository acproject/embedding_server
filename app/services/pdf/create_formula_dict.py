import os

def create_formula_dict(output_path):
    """
    创建公式识别字典文件
    
    Args:
        output_path: 输出路径
    """
    # 基本LaTeX符号和字符
    latex_symbols = [
        # 基本符号
        '+', '-', '=', '*', '/', '(', ')', '[', ']', '{', '}', '|', '\\', '^', '_', '&', '%', '$', '#', '@', '!', '?', ':', ';', ',', '.', '<', '>', '~',
        # 希腊字母
        '\\alpha', '\\beta', '\\gamma', '\\delta', '\\epsilon', '\\varepsilon', '\\zeta', '\\eta', '\\theta', '\\vartheta', '\\iota', '\\kappa', '\\lambda', '\\mu', '\\nu', '\\xi', '\\pi', '\\varpi', '\\rho', '\\varrho', '\\sigma', '\\varsigma', '\\tau', '\\upsilon', '\\phi', '\\varphi', '\\chi', '\\psi', '\\omega',
        '\\Gamma', '\\Delta', '\\Theta', '\\Lambda', '\\Xi', '\\Pi', '\\Sigma', '\\Upsilon', '\\Phi', '\\Psi', '\\Omega',
        # 数学运算符
        '\\sum', '\\prod', '\\coprod', '\\int', '\\oint', '\\bigcap', '\\bigcup', '\\bigsqcup', '\\bigvee', '\\bigwedge', '\\bigodot', '\\bigotimes', '\\bigoplus', '\\biguplus',
        # 关系符号
        '\\leq', '\\geq', '\\equiv', '\\models', '\\prec', '\\succ', '\\sim', '\\perp', '\\preceq', '\\succeq', '\\simeq', '\\mid', '\\ll', '\\gg', '\\asymp', '\\parallel', '\\subset', '\\supset', '\\approx', '\\bowtie', '\\subseteq', '\\supseteq', '\\cong', '\\Join', '\\sqsubset', '\\sqsupset', '\\neq', '\\smile', '\\sqsubseteq', '\\sqsupseteq', '\\doteq', '\\frown', '\\in', '\\ni', '\\propto', '\\vdash', '\\dashv',
        # 其他常用符号
        '\\infty', '\\nabla', '\\partial', '\\forall', '\\exists', '\\neg', '\\emptyset', '\\Re', '\\Im', '\\top', '\\bot', '\\angle', '\\triangle', '\\backslash', '\\prime', '\\ldots', '\\cdots', '\\vdots', '\\ddots',
        # 数字和字母
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        # 常用函数名
        '\\sin', '\\cos', '\\tan', '\\cot', '\\sec', '\\csc', '\\arcsin', '\\arccos', '\\arctan', '\\sinh', '\\cosh', '\\tanh', '\\coth', '\\log', '\\ln', '\\exp', '\\lim', '\\limsup', '\\liminf', '\\max', '\\min', '\\sup', '\\inf', '\\det', '\\dim', '\\mod', '\\gcd', '\\hom', '\\ker', '\\Pr', '\\arg',
        # 常用环境
        '\\begin', '\\end', 'matrix', 'pmatrix', 'bmatrix', 'vmatrix', 'Vmatrix', 'array', 'align', 'equation', 'gather', 'cases',
        # 空格
        '\\quad', '\\qquad', '\\,', '\\:', '\\;', '\\!', '~',
        # 其他常用命令
        '\\frac', '\\sqrt', '\\overline', '\\underline', '\\widehat', '\\widetilde', '\\overleftarrow', '\\overrightarrow', '\\overbrace', '\\underbrace', '\\not', '\\left', '\\right', '\\big', '\\Big', '\\bigg', '\\Bigg', '\\mathbb', '\\mathbf', '\\mathcal', '\\mathrm', '\\mathit', '\\mathsf', '\\mathtt', '\\text', '\\textbf', '\\textit'
    ]
    
    # 添加更多常见的数学符号和命令
    # 这里只是一个基本集合，实际的字典可能需要更多符号
    
    # 写入字典文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, symbol in enumerate(latex_symbols):
            f.write(f"{symbol}\n")
    
    print(f"公式字典文件已创建: {output_path}")
    print(f"包含 {len(latex_symbols)} 个基本LaTeX符号")

if __name__ == "__main__":
    # 获取模型目录
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
    formula_dir = os.path.join(models_dir, "formula")
    
    # 确保目录存在
    os.makedirs(formula_dir, exist_ok=True)
    
    # 创建字典文件
    dict_path = os.path.join(formula_dir, "formula_dict.txt")
    create_formula_dict(dict_path)