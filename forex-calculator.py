import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import math

# Configuración de la página
st.set_page_config(
    page_title="Calculadora de Posición Forex",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar estado de la sesión si es necesario
if 'sim_counter' not in st.session_state:
    st.session_state.sim_counter = 0

# Estilos personalizados
st.markdown("""
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
""", unsafe_allow_html=True)

# Título principal
st.markdown('<div class="main-header">🌐 Calculadora de Posición para Forex</div>', unsafe_allow_html=True)

# ======= SECCIÓN DE PARÁMETROS DE ENTRADA =======
st.markdown('<div class="sub-header">📊 Parámetros de Entrada</div>', unsafe_allow_html=True)

col_param1, col_param2 = st.columns(2)

with col_param1:
    # Parámetros básicos
    balance = st.number_input('Balance de la cuenta ($)', min_value=10.0, max_value=1000000.0, value=200000.0,
                              step=100.0)
    riesgo = st.slider('Riesgo (%)', min_value=0.1, max_value=10.0, value=1.0, step=1.0)
    comision = st.number_input('Comisión por lote ($)', min_value=0.0, max_value=100.0, value=4.0, step=1.0)
    stop_loss = st.number_input('Stop Loss (pips)', min_value=0.1, max_value=1000.0, value=50.0, step=0.1,
                                format="%.1f")

with col_param2:
    # Selector de par de divisas
    par_divisa = st.selectbox(
        'Par de divisas',
        ['EUR/USD', 'GBP/USD', 'XAU/USD'],
        index=0
    )

    # Valor del pip por lote estándar (100,000 unidades)
    pip_values = {
        'EUR/USD': 10.0,
        'GBP/USD': 10.0,
        'XAU/USD': 100.0  # Oro tiene un valor de pip diferente (10 USD para movimiento de 0.01)
    }

    pip_value = pip_values[par_divisa]

    # Añadir información sobre el tamaño del pip para el par seleccionado
    if par_divisa == 'XAU/USD':
        st.markdown(
            f'<div class="info-text">Para {par_divisa}, 1 pip = 0.01 (10 centavos). Valor por lote estándar: ${pip_values[par_divisa]:.2f}</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="info-text">Para {par_divisa}, 1 pip = 0.0001 (1 punto). Valor por lote estándar: ${pip_values[par_divisa]:.2f}</div>',
            unsafe_allow_html=True)

    # Tamaño de lote personalizado
    lot_size_options = st.radio(
        "Selecciona el tipo de lote",
        ["Estándar (100,000)", "Mini (10,000)", "Micro (1,000)"],
        horizontal=True
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

    # Relación riesgo:recompensa personalizable
    risk_reward_ratio = st.slider('Relación Riesgo:Recompensa', min_value=0.5, max_value=10.0, value=10.0, step=0.1)

    # Precio actual (opcional)
    use_price = st.checkbox('Especificar precio actual', value=False)
    if use_price:
        default_price = 1800.0 if par_divisa == 'XAU/USD' else 1.10
        step_value = 0.01 if par_divisa == 'XAU/USD' else 0.0001
        format_value = "%.2f" if par_divisa == 'XAU/USD' else "%.5f"
        current_price = st.number_input('Precio actual', min_value=0.1, value=default_price, step=step_value,
                                        format=format_value)

# Cálculos
monto_riesgo_target = balance * (riesgo / 100)  # El monto exacto que queremos arriesgar

# Cálculo correcto considerando la comisión por lote desde el inicio
# Resolviendo la ecuación: (lotes * valor_pip_loss) + (lotes * comision) = monto_riesgo_target
# Lo que se simplifica a: lotes = monto_riesgo_target / (valor_pip_loss + comision)
valor_pip_loss = stop_loss * pip_value

# Verificar si hay comisión y manejar los casos adecuadamente
if comision > 0:
    lotes_exactos = monto_riesgo_target / (valor_pip_loss + comision)
else:
    lotes_exactos = monto_riesgo_target / valor_pip_loss

# No redondeamos al tamaño de lote permitido, permitimos cualquier valor decimal
# Mantenemos la precisión completa del cálculo exacto
if lot_multiplier == 0.01:  # Para micro lotes, permitimos 3 decimales
    lotes_ajustados = math.floor(lotes_exactos * 1000) / 1000
else:  # Para mini y estándar, redondeamos a 2 decimales
    lotes_ajustados = math.floor(lotes_exactos * 100) / 100

# Cálculos finales con el tamaño de lote ajustado
riesgo_real = lotes_ajustados * valor_pip_loss + lotes_ajustados * comision
riesgo_real_porcentaje = (riesgo_real / balance) * 100

# ======= SECCIÓN DE RESULTADOS DEL CÁLCULO =======
st.markdown('<div class="sub-header">📈 Resultados del Cálculo</div>', unsafe_allow_html=True)

# Card para el tamaño de posición
col_result1, col_result2, col_result3 = st.columns([2, 1, 1])

with col_result1:
    st.markdown('<div class="card-title">Tamaño de Posición</div>', unsafe_allow_html=True)
    if lot_multiplier == 0.01:  # Para micro lotes, mostrar 3 decimales
        st.markdown(f'<div class="result-value">{lotes_ajustados:.3f} {lot_name}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-value">{lotes_ajustados:.2f} {lot_name}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-text">Equivalente a {lotes_ajustados * 100000 * lot_multiplier:,.0f} unidades</div>',
                unsafe_allow_html=True)

    # Mostrar información sobre el cálculo
    monto_riesgo_target = balance * (riesgo / 100)
    diferencia = abs(riesgo_real - monto_riesgo_target)
    precision = (1 - diferencia / monto_riesgo_target) * 100 if monto_riesgo_target > 0 else 100

    st.markdown(f'''<div class="info-text">
        <span class="highlight">Desglose del cálculo:</span><br>
        • Monto de riesgo objetivo: ${monto_riesgo_target:.2f}<br>
        • Pérdida por SL: ${lotes_ajustados * valor_pip_loss:.2f}<br>
        • Comisión: ${lotes_ajustados * comision:.2f}<br>
        • Riesgo total: ${riesgo_real:.2f} ({riesgo_real_porcentaje:.2f}% del balance)
    </div>''', unsafe_allow_html=True)

with col_result2:
    st.markdown('<div class="card-title">Monto en Riesgo</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="result-value loss">${riesgo_real:.2f}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-text">Equivalente al {riesgo_real_porcentaje:.2f}% del balance</div>',
                unsafe_allow_html=True)

with col_result3:
    # Calcular valor potencial por unidad de recompensa usando el valor dinámico
    reward = riesgo_real * risk_reward_ratio
    reward_percentage = riesgo_real_porcentaje * risk_reward_ratio

    st.markdown(f'<div class="card-title">Objetivo ({risk_reward_ratio:.1f}:1)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="result-value profit">${reward:.2f}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-text">Equivalente al {reward_percentage:.2f}% del balance</div>',
                unsafe_allow_html=True)

# Niveles de Precio (si se especificó un precio actual)
if use_price:
    st.markdown("---")
    entry_price = current_price
    if par_divisa.startswith('USD/'):
        # Para pares con USD como divisa base
        pip_factor = 0.01 if par_divisa == 'USD/JPY' else 0.0001
        sl_price = entry_price - (stop_loss * pip_factor)
        tp_price = entry_price + (stop_loss * pip_factor * risk_reward_ratio)
    elif par_divisa == 'XAU/USD':
        # Para oro (XAU/USD)
        pip_factor = 0.01  # En oro, 1 pip es 0.01
        sl_price = entry_price - (stop_loss * pip_factor)
        tp_price = entry_price + (stop_loss * pip_factor * risk_reward_ratio)
    else:
        # Para pares con USD como divisa cotizada
        pip_factor = 0.0001
        sl_price = entry_price - (stop_loss * pip_factor)
        tp_price = entry_price + (stop_loss * pip_factor * risk_reward_ratio)

    col_price1, col_price2, col_price3 = st.columns(3)

    with col_price1:
        st.markdown('<div class="card-title">Nivel de Entrada</div>', unsafe_allow_html=True)
        if par_divisa == 'XAU/USD':
            st.markdown(f'<div class="result-value highlight">{entry_price:.2f}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-value highlight">{entry_price:.5f}</div>', unsafe_allow_html=True)

    with col_price2:
        st.markdown('<div class="card-title">Nivel de Stop Loss</div>', unsafe_allow_html=True)
        if par_divisa == 'XAU/USD':
            st.markdown(f'<div class="result-value loss">{sl_price:.2f}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-value loss">{sl_price:.5f}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-text">({stop_loss} pips de distancia)</div>', unsafe_allow_html=True)

    with col_price3:
        st.markdown('<div class="card-title">Nivel de Take Profit</div>', unsafe_allow_html=True)
        if par_divisa == 'XAU/USD':
            st.markdown(f'<div class="result-value profit">{tp_price:.2f}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-value profit">{tp_price:.5f}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-text">({stop_loss * risk_reward_ratio} pips de distancia)</div>',
                    unsafe_allow_html=True)

# ======= SECCIÓN DE VISUALIZACIÓN =======
st.markdown('<div class="sub-header">📊 Visualización de Riesgo y Recompensa</div>', unsafe_allow_html=True)

col_viz1, col_viz2 = st.columns([2, 1])

with col_viz1:
    # Crear datos para el gráfico de barras
    labels = ['Balance', 'Monto en Riesgo', 'Objetivo de Ganancia']
    values = [balance, riesgo_real, reward]
    colors = ['#17a2b8', '#dc3545', '#28a745']

    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors
        )
    ])

    fig.update_layout(
        title='Comparación de Fondos',
        xaxis_title='',
        yaxis_title='Monto ($)',
        height=400,
        template='plotly_white',
        margin=dict(l=10, r=10, t=40, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

with col_viz2:
    # Gráfico de pastel para visualizar el porcentaje de riesgo
    labels = ['Riesgo', 'Balance Restante']
    values = [riesgo_real, balance - riesgo_real]
    colors = ['#dc3545', '#e9ecef']

    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            hole=0.7,
            marker_colors=colors
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
        height=300
    )

    st.plotly_chart(fig, use_container_width=True)

# ======= SECCIÓN DE SIMULADOR =======
st.markdown('<div class="sub-header">🎮 Simulador de Escenarios</div>', unsafe_allow_html=True)

# Configuración del simulador
col_sim1, col_sim2 = st.columns([1, 2])

with col_sim1:
    st.markdown('<div class="card-title">Configure su simulación</div>', unsafe_allow_html=True)

    num_trades = st.slider('Número de operaciones', min_value=1, max_value=50, value=10)
    win_rate = st.slider('Tasa de éxito (%)', min_value=10, max_value=90, value=50)

    # Usar la relación de riesgo:recompensa definida anteriormente
    st.markdown(f'<div class="info-text">Utilizando relación R:R de {risk_reward_ratio:.1f}:1</div>',
                unsafe_allow_html=True)

    # Añadir botón para ejecutar una nueva simulación con los mismos parámetros
    if st.button('Generar Nueva Simulación', key='new_sim'):
        st.session_state.sim_counter = st.session_state.get('sim_counter', 0) + 1

# Calcular resultados de simulación completamente aleatorios
# Generar una nueva semilla cada vez para garantizar que los resultados sean diferentes
np.random.seed(int(datetime.now().timestamp()) + st.session_state.sim_counter)

results = []
balance_acum = balance

# Generar un arreglo de resultados basado exactamente en la tasa de éxito
# Esto garantiza que el porcentaje de operaciones ganadoras sea exactamente el especificado
result_array = ['Ganancia'] * int(num_trades * win_rate / 100) + ['Pérdida'] * (
            num_trades - int(num_trades * win_rate / 100))
# Mezclar el arreglo para obtener una secuencia aleatoria
np.random.shuffle(result_array)

for i in range(num_trades):
    result = result_array[i]

    if result == 'Ganancia':
        # Para las ganancias, añadimos algo de variabilidad al profit
        # entre 0.8 y 1.2 veces el objetivo de R:R
        profit_multiplier = 0.8 + np.random.random() * 0.4  # entre 0.8 y 1.2
        profit = reward * profit_multiplier
    else:
        # Para las pérdidas, también añadimos variabilidad
        # entre 0.9 y 1.1 veces el riesgo (las pérdidas tienden a ser más precisas)
        loss_multiplier = 0.9 + np.random.random() * 0.2  # entre 0.9 y 1.1
        profit = -riesgo_real * loss_multiplier

    balance_acum += profit

    results.append({
        'Operación': i + 1,
        'Resultado': result,
        'P&L': profit,
        'Balance': balance_acum
    })

# Calcular estadísticas finales
final_balance = results[-1]['Balance'] if results else balance
profit_loss = final_balance - balance
percent_change = (profit_loss / balance) * 100

# Calcular métricas adicionales
total_wins = sum(1 for r in results if r['Resultado'] == 'Ganancia')
total_losses = num_trades - total_wins
avg_win = sum(r['P&L'] for r in results if r['Resultado'] == 'Ganancia') / total_wins if total_wins > 0 else 0
avg_loss = sum(abs(r['P&L']) for r in results if r['Resultado'] == 'Pérdida') / total_losses if total_losses > 0 else 0

# Drawdown máximo
peak_balance = balance
max_drawdown = 0
max_drawdown_pct = 0
current_drawdown = 0

for r in results:
    if r['Balance'] > peak_balance:
        peak_balance = r['Balance']
        current_drawdown = 0
    else:
        current_drawdown = peak_balance - r['Balance']
        current_drawdown_pct = (current_drawdown / peak_balance) * 100

        if current_drawdown > max_drawdown:
            max_drawdown = current_drawdown
            max_drawdown_pct = current_drawdown_pct

# Tabla y Gráfico de resultados
col_sim_result1, col_sim_result2 = st.columns([1, 1])

with col_sim_result1:
    st.markdown('<div class="card-title">Detalle de operaciones</div>', unsafe_allow_html=True)

    df_results = pd.DataFrame(results)


    # Aplicar estilo condicional
    def highlight_result(val):
        if val == 'Ganancia':
            return 'color: #28a745; font-weight: bold'
        elif val == 'Pérdida':
            return 'color: #dc3545; font-weight: bold'
        else:
            return ''


    def highlight_pnl(val):
        if val > 0:
            return 'color: #28a745; font-weight: bold'
        elif val < 0:
            return 'color: #dc3545; font-weight: bold'
        else:
            return ''


    # Aplicar estilos y formateos
    styled_df = df_results.style.map(highlight_result, subset=['Resultado']) \
        .map(highlight_pnl, subset=['P&L']) \
        .format({'P&L': '${:.2f}', 'Balance': '${:.2f}'})

    # Mostrar dataframe ajustado a la columna
    st.dataframe(styled_df, height=300, use_container_width=True)

with col_sim_result2:
    st.markdown('<div class="card-title">Evolución del Balance</div>', unsafe_allow_html=True)

    # Gráfico de línea para mostrar la evolución del balance
    balances = [balance] + [r['Balance'] for r in results]
    trades = list(range(len(balances)))

    fig = go.Figure()

    # Añadir línea de balance inicial
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

    # Añadir la línea principal
    fig.add_trace(
        go.Scatter(
            x=trades,
            y=balances,
            mode='lines+markers',
            name='Balance',
            line=dict(color='#0078ff', width=3),
            marker=dict(size=8)
        )
    )

    fig.update_layout(
        xaxis_title='Número de Operaciones',
        yaxis_title='Balance ($)',
        height=300,
        template='plotly_white',
        margin=dict(l=10, r=10, t=10, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

# Mostrar resultados de la simulación
col_sim_stats1, col_sim_stats2 = st.columns(2)

with col_sim_stats1:
    st.markdown('<div class="card-title">Resultado Final</div>', unsafe_allow_html=True)
    profit_loss_color = "profit" if profit_loss >= 0 else "loss"
    profit_loss_sign = "+" if profit_loss >= 0 else ""
    st.markdown(f'<div class="result-value {profit_loss_color}">{profit_loss_sign}${profit_loss:.2f}</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="info-text">{profit_loss_sign}{percent_change:.2f}% del balance inicial</div>',
                unsafe_allow_html=True)

    # Mostrar drawdown máximo
    st.markdown(
        f'<div class="info-text">Drawdown máximo: <span class="loss">${max_drawdown:.2f}</span> ({max_drawdown_pct:.2f}%)</div>',
        unsafe_allow_html=True)

    # Mostrar promedio de ganancias/pérdidas
    if total_wins > 0:
        st.markdown(f'<div class="info-text">Ganancia promedio: <span class="profit">${avg_win:.2f}</span></div>',
                    unsafe_allow_html=True)
    if total_losses > 0:
        st.markdown(f'<div class="info-text">Pérdida promedio: <span class="loss">${avg_loss:.2f}</span></div>',
                    unsafe_allow_html=True)

with col_sim_stats2:
    st.markdown('<div class="card-title">Operaciones</div>', unsafe_allow_html=True)
    actual_wins = sum(1 for r in results if r['Resultado'] == 'Ganancia')
    actual_win_rate = (actual_wins / num_trades) * 100
    st.markdown(f'<div class="result-value">{actual_wins}/{num_trades} exitosas</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-text">Tasa de éxito real: {actual_win_rate:.1f}% (objetivo: {win_rate:.1f}%)</div>',
                unsafe_allow_html=True)

    # Calcular racha más larga de operaciones ganadoras y perdedoras
    current_streak = 1
    max_win_streak = 0
    max_loss_streak = 0
    current_type = results[0]['Resultado'] if results else None

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

    # No olvidar la última racha
    if current_type == 'Ganancia':
        max_win_streak = max(max_win_streak, current_streak)
    else:
        max_loss_streak = max(max_loss_streak, current_streak)

    st.markdown(
        f'<div class="info-text">Racha más larga de ganancias: <span class="profit">{max_win_streak}</span></div>',
        unsafe_allow_html=True)
    st.markdown(
        f'<div class="info-text">Racha más larga de pérdidas: <span class="loss">{max_loss_streak}</span></div>',
        unsafe_allow_html=True)

# Pie de página
st.markdown('---')
st.markdown(
    f'<div style="text-align: center; color: #6c757d;">Calculadora de Posición Forex • {datetime.now().strftime("%Y-%m-%d")}</div>',
    unsafe_allow_html=True)
st.markdown(
    '<div style="text-align: center; color: #6c757d; font-size: 12px;">Esta calculadora es solo para fines educativos. Trading con divisas conlleva riesgos significativos.</div>',
    unsafe_allow_html=True)