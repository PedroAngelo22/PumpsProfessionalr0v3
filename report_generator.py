# report_generator.py

from fpdf import FPDF
from datetime import datetime
import io

class PDFReport(FPDF):
    def __init__(self, project_name, scenario_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project_name = project_name
        self.scenario_name = scenario_name
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        """ Cria o cabeçalho do relatório em cada página. """
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Relatório de Análise Hidráulica', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 6, f'Projeto: {self.project_name}', 0, 1, 'C')
        self.cell(0, 6, f'Cenário: {self.scenario_name}', 0, 1, 'C')
        self.ln(10) # Pula uma linha

    def footer(self):
        """ Cria o rodapé do relatório em cada página. """
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')
        self.set_x(10) # Posiciona à esquerda
        self.cell(0, 10, f'Gerado em: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}', 0, 0, 'L')


    def add_section_title(self, title):
        """ Adiciona um título de seção formatado. """
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(230, 230, 230) # Cinza claro
        self.cell(0, 8, title, 0, 1, 'L', fill=True)
        self.ln(4)

    def add_key_value_table(self, data_dict):
        """ Cria uma tabela simples de Chave-Valor. """
        self.set_font('Arial', '', 10)
        for key, value in data_dict.items():
            self.set_font('', 'B')
            self.cell(60, 8, f'{key}:', border=1)
            self.set_font('', '')
            self.cell(0, 8, f' {value}', border=1)
            self.ln()
        self.ln(5)

    def add_results_metrics(self, metrics_data):
        """ Adiciona as métricas de resultado em colunas. """
        self.set_font('Arial', 'B', 11)
        
        num_metrics = len(metrics_data)
        cell_width = 190 / num_metrics if num_metrics > 0 else 190

        for title, _ in metrics_data:
            self.cell(cell_width, 7, title, 1, 0, 'C')
        self.ln()

        self.set_font('', '')
        for _, value in metrics_data:
            self.cell(cell_width, 10, value, 1, 0, 'C')
        self.ln()
        self.ln(5)

    def add_matplotlib_chart(self, chart_figure):
        """ Salva uma figura Matplotlib em memória e a insere no PDF. """
        chart_buffer = io.BytesIO()
        chart_figure.savefig(chart_buffer, format='PNG', dpi=300)
        chart_buffer.seek(0)
        
        self.image(chart_buffer, x=self.get_x(), y=self.get_y(), w=190)
        chart_buffer.close()
        self.ln(5)

# --- Função principal que será chamada pelo app.py ---
def generate_report(project_name, scenario_name, params_data, results_data, metrics_data, chart_figure):
    """
    Função principal que orquestra a criação do PDF.
    Recebe todos os dados necessários e retorna o PDF como bytes.
    """
    pdf = PDFReport(project_name, scenario_name)
    pdf.add_page()
    
    # Seção 1: Parâmetros Gerais
    pdf.add_section_title('Parâmetros Gerais da Simulação')
    pdf.add_key_value_table(params_data)

    # Seção 2: Resultados Principais
    pdf.add_section_title('Resultados no Ponto de Operação')
    pdf.add_results_metrics(metrics_data)
    
    # Seção 3: Detalhes do Custo Energético
    pdf.add_section_title('Análise de Custo Energético')
    pdf.add_key_value_table(results_data)

    # Seção 4: Gráfico das Curvas
    pdf.add_section_title('Gráfico: Curva da Bomba vs. Curva do Sistema')
    pdf.add_matplotlib_chart(chart_figure)

    # CORREÇÃO FINAL: Converte explicitamente a saída para 'bytes'
    return bytes(pdf.output())
