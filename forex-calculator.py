import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from decimal import Decimal, getcontext, ROUND_HALF_UP
import math

# Configurar precisión decimal para cálculos financieros
getcontext().prec = 10
getcontext().rounding = ROUND_HALF_UP

# Configuración de la página
st.set_page_config(
    page_title="Calculadora de Posición Forex",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.streamlit.io',
        'About': "Calculadora Forex v2.0 - Actualizado 2025"
    }
)

# Inicializar estado de la sesión
if 'sim_counter' not in st.session_state:
    st.session_state.sim_counter = 0
if 'last_calculation' not in st.session_state:
    st.session_state.last_calculation = None
if 'last_sim_seed' not in st.session_state:
    st.session_state.last_sim_seed = None

# Estilos personalizados usando st.html (método moderno)
st.html("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #0078ff;
        text-align: center;
        margin-bottom: 20px;
        padding-bottom: 15px;
        border-bottom: 2px solid #f0f2f6;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #0078ff;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .card-title {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 15px;
        color: #0078ff;
    }
    .result-value {
        font-size: 24px;
        font-weight: bold;
        color: #0078ff;
    }
    .info-text {
        color: #6c757d;
        font-size: 14px;
    }
    .highlight {
        background-color: #e6f3ff;
        padding: 2px 5px;
        border-radius: 3px;
        font-weight: bold;
    }
    .profit {
        color: #28a745;
        font-weight: bold;
    }
    .loss {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""")

# Configuración de Plotly (definida globalmente)
PLOTLY_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'forex_chart',
        'height': 800,
        'width': 1200,
        'scale': 2
    }
}

# Funciones auxiliares con caché
@st.cache_data
def calcular_posicion(balance_float: float, riesgo_float: float, stop_loss_float: float, 
                      pip_value_float: float, comision_float: float) -> dict:
    """Calcula el tamaño de posición con precisión decimal"""
    # Convertir a Decimal para precisión financiera
    balance = Decimal(str(balance_float))
    riesgo = Decimal(str(riesgo_float))
    stop_loss = Decimal(str(stop_loss_float))
    pip_value = Decimal(str(pip_value_float))
    comision = Decimal(str(comision_float))
    
    monto_riesgo = balance * (riesgo / Decimal('100'))
    valor_pip_loss = stop_loss * pip_value
    
    if comision > 0:
        lotes = monto_riesgo / (valor_pip_loss + comision)
    else:
        lotes = monto_riesgo / valor_pip_loss
    
    riesgo_real = lotes * (valor_pip_loss + comision)
    
    return {
        'lotes': float(lotes),
        'riesgo_real': float(riesgo_real),
        'monto_riesgo_target': float(monto_riesgo),
        'valor_pip_loss': float(valor_pip_loss)
    }

def validar_parametros(balance: float, riesgo: float, stop_loss: float, comision: float) -> list:
    """Valida los parámetros de entrada"""
    errores = []
    
    if balance <= 0:
        errores.append("⚠️ El balance debe ser mayor a 0")
    if not (0.1 <= riesgo <= 5.0):
        errores.append("⚠️ El riesgo debe estar entre 0.1% y 5.0%")
    if stop_loss <= 0:
        errores.append("⚠️ El stop loss debe ser mayor a 0")
    if comision < 0:
        errores.append("⚠️ La comisión no puede ser negativa")
    
    return errores

@st.cache_data
def generar_simulacion(num_trades: int, win_rate: float, riesgo_real: float, 
                       reward: float, balance_inicial: float, seed: int = None) -> tuple:
    """Genera una simulación reproducible"""
    if seed is None:
        seed = int(datetime.now().timestamp() * 1000)
    
    np.random.seed(seed)
    
    results = []
    balance_acum = balance_inicial
    
    # Generar array de resultados basado en la tasa de éxito
    result_array = (['Ganancia'] * int(num_trades * win_rate / 100) + 
                   ['Pérdida'] * (num_trades - int(num_trades * win_rate / 100)))
    np.random.shuffle(result_array)
    
    for i in range(num_trades):
        result = result_array[i]
        
        if result == 'Ganancia':
            profit_multiplier = 0.8 + np.random.random() * 0.4
            profit = reward * profit_multiplier
        else:
            loss_multiplier = 0.9 + np.random.random() * 0.2
            profit = -riesgo_real * loss_multiplier
        
        balance_acum += profit
        
        results.append({
            'Operación': i + 1,
            'Resultado': result,
            'P&L': profit,
            'Balance': balance_acum
        })
    
    return results, seed

def calcular_estadisticas_simulacion(results: list, balance_inicial: float) -> dict:
    """Calcula estadísticas de la simulación"""
    if not results:
        return {}
    
    final_balance = results[-1]['Balance']
    profit_loss = final_balance - balance_inicial
    percent_change = (profit_loss / balance_inicial) * 100
    
    total_wins = sum(1 for r in results if r['Resultado'] == 'Ganancia')
    total_losses = len(results) - total_wins
    
    avg_win = (sum(r['P&L'] for r in results if r['Resultado'] == 'Ganancia') / total_wins 
               if total_wins > 0 else 0)
    avg_loss = (sum(abs(r['P&L']) for r in results if r['Resultado'] == 'Pérdida') / total_losses 
                if total_losses > 0 else 0)
    
    # Calcular drawdown máximo
    peak_balance = balance_inicial
    max_drawdown = 0
    max_drawdown_pct = 0
    
    for r in results:
        if r['Balance'] > peak_balance:
            peak_balance = r['Balance']
        else:
            current_drawdown = peak_balance - r['Balance']
            current_drawdown_pct = (current_drawdown / peak_balance) * 100
            
            if current_drawdown > max_drawdown:
                max_drawdown = current_drawdown
                max_drawdown_pct = current_drawdown_pct
    
    # Calcular rachas
    current_streak = 1
    max_win_streak = 0
    max_loss_streak = 0
    current_type = results[0]['Resultado']
    
    for i in range(1, len(results)):
        if results[i]['Resultado'] == current_type:
            current_streak += 1
        else:
            if current_type == 'Ganancia':
                max_win_streak = max(max_win_streak, current_streak)
            else:
                max_loss_streak = max(max_loss_streak, current_streak)
            current_streak = 1
            current_type = results[i]['Resultado']
    
    if current_type == 'Ganancia':
        max_win_streak = max(max_win_streak, current_streak)
    else:
        max_loss_streak = max(max_loss_streak, current_streak)
    
    actual_win_rate = (total_wins / len(results)) * 100
    
    return {
        'final_balance': final_balance,
        'profit_loss': profit_loss,
        'percent_change': percent_change,
        'total_wins': total_wins,
        'total_losses': total_losses,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,
        'max_win_streak': max_win_streak,
        'max_loss_streak': max_loss_streak,
        'actual_win_rate': actual_win_rate
    }

# Título principal
st.html('<div class="main-header">🌐 Calculadora de Posición para Forex</div>')

# ======= SECCIÓN DE PARÁMETROS DE ENTRADA =======
st.html('<div class="sub-header">📊 Parámetros de Entrada</div>')

col_param1, col_param2 = st.columns(2)

with col_param1:
    balance = st.number_input(
        'Balance de la cuenta ($)', 
        min_value=10.0, 
        max_value=1000000.0, 
        value=100000.0,
        step=100.0,
        help="El capital total disponible en tu cuenta"
    )
    
    riesgo = st.slider(
        'Riesgo (%)', 
        min_value=0.1, 
        max_value=5.0, 
        value=1.0, 
        step=0.01,
        help="Porcentaje del balance que estás dispuesto a arriesgar"
    )
    
    comision = st.number_input(
        'Comisión por lote ($)', 
        min_value=0.0, 
        max_value=100.0, 
        value=4.0, 
        step=1.0,
        help="Comisión que cobra el broker por lote"
    )
    
    stop_loss = st.number_input(
        'Stop Loss (pips)', 
        min_value=0.1, 
        max_value=1000.0, 
        value=5.0, 
        step=0.1,
        format="%.1f",
        help="Distancia en pips hasta tu stop loss"
    )

with col_param2:
    par_divisa = st.selectbox(
        'Par de divisas',
        ['XAU/USD', 'EUR/USD', 'GBP/USD'],
        index=0,
        help="Selecciona el par de divisas a operar"
    )
    
    # Valor del pip por lote estándar
    pip_values = {
        'EUR/USD': 10.0,
        'GBP/USD': 10.0,
        'XAU/USD': 100.0
    }
    
    pip_value = pip_values[par_divisa]
    
    # Información sobre el pip
    if par_divisa == 'XAU/USD':
        st.html(
            f'<div class="info-text">Para {par_divisa}, 1 pip = 0.01 (10 centavos). '
            f'Valor por lote estándar: ${pip_values[par_divisa]:.2f}</div>'
        )
    else:
        st.html(
            f'<div class="info-text">Para {par_divisa}, 1 pip = 0.0001 (1 punto). '
            f'Valor por lote estándar: ${pip_values[par_divisa]:.2f}</div>'
        )
    
    # Tamaño de lote
    lot_size_options = st.radio(
        "Selecciona el tipo de lote",
        ["Estándar (100,000)", "Mini (10,000)", "Micro (1,000)"],
        horizontal=True,
        help="Tamaño del contrato a operar"
    )
    
    if lot_size_options == "Estándar (100,000)":
        lot_multiplier = 1.0
        lot_name = "lotes estándar"
    elif lot_size_options == "Mini (10,000)":
        lot_multiplier = 0.1
        lot_name = "mini lotes"
        pip_value *= 0.1
    else:
        lot_multiplier = 0.01
        lot_name = "micro lotes"
        pip_value *= 0.01
    
    # Relación riesgo:recompensa
    risk_reward_ratio = st.slider(
        'Relación Riesgo:Recompensa', 
        min_value=0.5, 
        max_value=10.0, 
        value=3.0, 
        step=0.5,
        help="Cuántas veces tu riesgo esperas ganar"
    )
    
    # Precio actual
    use_price = st.checkbox(
        'Especificar precio actual', 
        value=False,
        help="Activar para calcular niveles de entrada, SL y TP"
    )
    
    if use_price:
        default_price = 1800.0 if par_divisa == 'XAU/USD' else 1.10
        step_value = 0.01 if par_divisa == 'XAU/USD' else 0.0001
        format_value = "%.2f" if par_divisa == 'XAU/USD' else "%.5f"
        current_price = st.number_input(
            'Precio actual', 
            min_value=0.1, 
            value=default_price, 
            step=step_value,
            format=format_value
        )

# Validar parámetros
errores = validar_parametros(balance, riesgo, stop_loss, comision)
if errores:
    for error in errores:
        st.error(error)
    st.stop()

# Cálculos con manejo de errores
try:
    calc_result = calcular_posicion(balance, riesgo, stop_loss, pip_value, comision)
    
    monto_riesgo_target = calc_result['monto_riesgo_target']
    lotes_exactos = calc_result['lotes']
    riesgo_real = calc_result['riesgo_real']
    valor_pip_loss = calc_result['valor_pip_loss']
    
    # Ajustar lotes según el multiplicador
    if lot_multiplier == 0.01:
        lotes_ajustados = math.floor(lotes_exactos * 1000) / 1000
    else:
        lotes_ajustados = math.floor(lotes_exactos * 100) / 100
    
    # Recalcular con lotes ajustados
    riesgo_real = lotes_ajustados * valor_pip_loss + lotes_ajustados * comision
    riesgo_real_porcentaje = (riesgo_real / balance) * 100
    
    # Guardar en session state
    st.session_state.last_calculation = {
        'lotes': lotes_ajustados,
        'riesgo': riesgo_real,
        'timestamp': datetime.now()
    }
    
except ZeroDivisionError:
    st.error("❌ Error: El valor del pip y la comisión no pueden ser ambos cero")
    st.stop()
except Exception as e:
    st.error(f"❌ Error inesperado en el cálculo: {str(e)}")
    st.stop()

# ======= SECCIÓN DE RESULTADOS DEL CÁLCULO =======
st.html('<div class="sub-header">📈 Resultados del Cálculo</div>')

col_result1, col_result2, col_result3 = st.columns([2, 1, 1])

with col_result1:
    st.html('<div class="card-title">Tamaño de Posición</div>')
    
    # Usar st.metric para mejor visualización
    lotes_display = f"{lotes_ajustados:.3f}" if lot_multiplier == 0.01 else f"{lotes_ajustados:.2f}"
    diferencia = lotes_ajustados - lotes_exactos
    
    st.metric(
        label=f"Tamaño ({lot_name})",
        value=lotes_display,
        delta=f"{diferencia:.4f}",
        delta_color="off"
    )
    
    st.html(
        f'<div class="info-text">Equivalente a {lotes_ajustados * 100000 * lot_multiplier:,.0f} unidades</div>'
    )
    
    # Desglose del cálculo
    diferencia_monto = abs(riesgo_real - monto_riesgo_target)
    precision = (1 - diferencia_monto / monto_riesgo_target) * 100 if monto_riesgo_target > 0 else 100
    
    st.html(f'''<div class="info-text">
        <span class="highlight">Desglose del cálculo:</span><br>
        • Monto de riesgo objetivo: ${monto_riesgo_target:.2f}<br>
        • Pérdida por SL: ${lotes_ajustados * valor_pip_loss:.2f}<br>
        • Comisión: ${lotes_ajustados * comision:.2f}<br>
        • Riesgo total: ${riesgo_real:.2f} ({riesgo_real_porcentaje:.2f}% del balance)<br>
        • Precisión del cálculo: {precision:.2f}%
    </div>''')

with col_result2:
    st.html('<div class="card-title">Monto en Riesgo</div>')
    
    st.metric(
        label="Riesgo Total",
        value=f"${riesgo_real:.2f}",
        delta=f"{riesgo_real_porcentaje:.2f}% del balance",
        delta_color="inverse"
    )

with col_result3:
    reward = riesgo_real * risk_reward_ratio
    reward_percentage = riesgo_real_porcentaje * risk_reward_ratio
    
    st.html(f'<div class="card-title">Objetivo ({risk_reward_ratio:.1f}:1)</div>')
    
    st.metric(
        label="Ganancia Potencial",
        value=f"${reward:.2f}",
        delta=f"{reward_percentage:.2f}% del balance",
        delta_color="normal"
    )

# Niveles de Precio
if use_price:
    st.divider()
    
    entry_price = current_price
    
    if par_divisa == 'XAU/USD':
        pip_factor = 0.01
    elif par_divisa.startswith('USD/'):
        pip_factor = 0.01 if par_divisa == 'USD/JPY' else 0.0001
    else:
        pip_factor = 0.0001
    
    sl_price = entry_price - (stop_loss * pip_factor)
    tp_price = entry_price + (stop_loss * pip_factor * risk_reward_ratio)
    
    col_price1, col_price2, col_price3 = st.columns(3)
    
    with col_price1:
        st.html('<div class="card-title">Nivel de Entrada</div>')
        if par_divisa == 'XAU/USD':
            st.html(f'<div class="result-value highlight">{entry_price:.2f}</div>')
        else:
            st.html(f'<div class="result-value highlight">{entry_price:.5f}</div>')
    
    with col_price2:
        st.html('<div class="card-title">Nivel de Stop Loss</div>')
        if par_divisa == 'XAU/USD':
            st.html(f'<div class="result-value loss">{sl_price:.2f}</div>')
        else:
            st.html(f'<div class="result-value loss">{sl_price:.5f}</div>')
        st.html(f'<div class="info-text">({stop_loss} pips de distancia)</div>')
    
    with col_price3:
        st.html('<div class="card-title">Nivel de Take Profit</div>')
        if par_divisa == 'XAU/USD':
            st.html(f'<div class="result-value profit">{tp_price:.2f}</div>')
        else:
            st.html(f'<div class="result-value profit">{tp_price:.5f}</div>')
        st.html(f'<div class="info-text">({stop_loss * risk_reward_ratio:.1f} pips de distancia)</div>')

# ======= SECCIÓN DE VISUALIZACIÓN =======
st.html('<div class="sub-header">📊 Visualización de Riesgo y Recompensa</div>')

col_viz1, col_viz2 = st.columns([2, 1])

with col_viz1:
    labels = ['Balance', 'Monto en Riesgo', 'Objetivo de Ganancia']
    values = [balance, riesgo_real, reward]
    colors = ['#17a2b8', '#dc3545', '#28a745']
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f'${v:,.2f}' for v in values],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title='Comparación de Fondos',
        xaxis_title='',
        yaxis_title='Monto ($)',
        height=400,
        template='plotly_white',
        margin=dict(l=10, r=10, t=40, b=20),
        showlegend=False
    )
    
    st.plotly_chart(
        fig, 
        use_container_width=True,
        theme="streamlit",
        config=PLOTLY_CONFIG
    )

with col_viz2:
    labels = ['Riesgo', 'Balance Restante']
    values = [riesgo_real, balance - riesgo_real]
    colors = ['#dc3545', '#e9ecef']
    
    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            hole=0.7,
            marker_colors=colors,
            textinfo='percent',
            hovertemplate='%{label}: $%{value:,.2f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=f'Porcentaje de Riesgo: {riesgo_real_porcentaje:.2f}%',
        annotations=[
            dict(
                text=f'{riesgo_real_porcentaje:.2f}%',
                x=0.5, y=0.5,
                font_size=20,
                showarrow=False
            )
        ],
        height=300,
        margin=dict(l=10, r=10, t=40, b=20)
    )
    
    st.plotly_chart(
        fig, 
        use_container_width=True,
        theme="streamlit",
        config=PLOTLY_CONFIG
    )

# ======= SECCIÓN DE SIMULADOR =======
st.html('<div class="sub-header">🎮 Simulador de Escenarios</div>')

col_sim1, col_sim2 = st.columns([1, 2])

with col_sim1:
    st.html('<div class="card-title">Configure su simulación</div>')
    
    num_trades = st.slider(
        'Número de operaciones', 
        min_value=1, 
        max_value=50, 
        value=10,
        help="Cantidad de trades a simular"
    )
    
    win_rate = st.slider(
        'Tasa de éxito (%)', 
        min_value=10, 
        max_value=90, 
        value=30,
        help="Porcentaje de operaciones ganadoras"
    )
    
    st.html(
        f'<div class="info-text">Utilizando relación R:R de {risk_reward_ratio:.1f}:1</div>'
    )
    
    if st.button('🎲 Generar Nueva Simulación', key='new_sim', width='stretch'):
        st.session_state.sim_counter += 1
        st.toast('¡Simulación generada!', icon='✅')
        st.rerun()

# Generar simulación
results, seed = generar_simulacion(
    num_trades, 
    win_rate, 
    riesgo_real, 
    reward, 
    balance,
    seed=st.session_state.sim_counter
)

st.session_state.last_sim_seed = seed

# Calcular estadísticas
stats = calcular_estadisticas_simulacion(results, balance)

# Tabla y Gráfico de resultados
col_sim_result1, col_sim_result2 = st.columns([1, 1])

with col_sim_result1:
    st.html('<div class="card-title">Detalle de operaciones</div>')
    
    df_results = pd.DataFrame(results)
    
    # Funciones de estilo para colorear
    def color_resultado(val):
        """Colorea toda la celda según el resultado"""
        if val == 'Ganancia':
            return 'background-color: #d4edda; color: #155724; font-weight: bold; text-align: center; padding: 5px'
        elif val == 'Pérdida':
            return 'background-color: #f8d7da; color: #721c24; font-weight: bold; text-align: center; padding: 5px'
        return ''
    
    def color_pnl(val):
        """Colorea el texto del P&L"""
        if val > 0:
            return 'color: #28a745; font-weight: bold'
        elif val < 0:
            return 'color: #dc3545; font-weight: bold'
        return ''
    
    def color_balance(val):
        """Colorea el balance según si ganó o perdió respecto al inicial"""
        if val > balance:
            return 'color: #28a745'
        elif val < balance:
            return 'color: #dc3545'
        return ''
    
    # Aplicar todos los estilos
    styled_df = (df_results.style
                 .map(color_resultado, subset=['Resultado'])
                 .map(color_pnl, subset=['P&L'])
                 .map(color_balance, subset=['Balance'])
                 .format({
                     'P&L': '${:,.2f}',
                     'Balance': '${:,.2f}',
                     'Operación': '{:d}'
                 })
                 .set_properties(**{
                     'text-align': 'right'
                 }, subset=['P&L', 'Balance', 'Operación']))
    
    st.dataframe(
        styled_df,
        width='stretch',
        height=300
    )

with col_sim_result2:
    st.html('<div class="card-title">Evolución del Balance</div>')
    
    balances = [balance] + [r['Balance'] for r in results]
    trades = list(range(len(balances)))
    
    fig = go.Figure()
    
    # Línea de balance inicial
    fig.add_shape(
        type="line",
        x0=0,
        y0=balance,
        x1=num_trades,
        y1=balance,
        line=dict(
            color="gray",
            width=1,
            dash="dash",
        )
    )
    
    # Línea principal con colores según profit/loss
    colors_line = ['green' if b >= balance else 'red' for b in balances]
    
    fig.add_trace(
        go.Scatter(
            x=trades,
            y=balances,
            mode='lines+markers',
            name='Balance',
            line=dict(color='#0078ff', width=3),
            marker=dict(size=8, color=colors_line),
            hovertemplate='Operación %{x}<br>Balance: $%{y:,.2f}<extra></extra>'
        )
    )
    
    fig.update_layout(
        xaxis_title='Número de Operaciones',
        yaxis_title='Balance ($)',
        height=300,
        template='plotly_white',
        margin=dict(l=10, r=10, t=10, b=20),
        showlegend=False
    )
    
    st.plotly_chart(
        fig, 
        use_container_width=True,
        theme="streamlit",
        config=PLOTLY_CONFIG
    )

# Mostrar resultados de la simulación
st.divider()

col_sim_stats1, col_sim_stats2, col_sim_stats3 = st.columns(3)

with col_sim_stats1:
    st.html('<div class="card-title">Resultado Final</div>')
    
    st.metric(
        label="Ganancia/Pérdida",
        value=f"${stats['profit_loss']:.2f}",
        delta=f"{stats['percent_change']:.2f}%"
    )
    
    st.html(
        f'<div class="info-text">Drawdown máximo: '
        f'<span class="loss">${stats["max_drawdown"]:.2f}</span> '
        f'({stats["max_drawdown_pct"]:.2f}%)</div>'
    )

with col_sim_stats2:
    st.html('<div class="card-title">Estadísticas de Operaciones</div>')
    
    st.metric(
        label="Operaciones Exitosas",
        value=f"{stats['total_wins']}/{num_trades}",
        delta=f"Tasa real: {stats['actual_win_rate']:.1f}%"
    )
    
    if stats['total_wins'] > 0:
        st.html(
            f'<div class="info-text">Ganancia promedio: '
            f'<span class="profit">${stats["avg_win"]:.2f}</span></div>'
        )
    if stats['total_losses'] > 0:
        st.html(
            f'<div class="info-text">Pérdida promedio: '
            f'<span class="loss">${stats["avg_loss"]:.2f}</span></div>'
        )

with col_sim_stats3:
    st.html('<div class="card-title">Rachas</div>')
    
    st.metric(
        label="Racha más larga de ganancias",
        value=f"{stats['max_win_streak']} ops",
        delta_color="off"
    )
    
    st.metric(
        label="Racha más larga de pérdidas",
        value=f"{stats['max_loss_streak']} ops",
        delta_color="off"
    )

# Pie de página
st.divider()
st.html(
    f'<div style="text-align: center; color: #6c757d;">'
    f'Calculadora de Posición Forex v2.0 • {datetime.now().strftime("%Y-%m-%d %H:%M")}'
    f'</div>'
)
st.html(
    '<div style="text-align: center; color: #6c757d; font-size: 12px;">'
    'Esta calculadora es solo para fines educativos. Trading con divisas conlleva riesgos significativos.'
    '</div>'
)

# Información de debug (opcional, comentar en producción)
if st.checkbox('🔧 Mostrar información de debug', value=False):
    with st.expander("Información de Sesión"):
        st.json({
            'sim_counter': st.session_state.sim_counter,
            'last_sim_seed': st.session_state.last_sim_seed,
            'last_calculation': str(st.session_state.last_calculation) if st.session_state.last_calculation else None
        })
