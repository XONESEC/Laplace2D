# INPUT LIBRARY
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import time

# ============================================================
# DESCRIPTION
# ============================================================

st.header("**Heat-Conduction with Laplace**", divider="gray")
st.subheader("**Properties**")
st.write("In this section, the properties that will be used in the calculation are defined. The required properties are:")

st.markdown(""" 
* **BC Left** = Boundary condition value (°C for Dirichlet, W/m² for Neumann)

* **BC Right** = Boundary condition value (°C for Dirichlet, W/m² for Neumann)

* **BC Top** = Boundary condition value (°C for Dirichlet, W/m² for Neumann)

* **BC Bottom** = Boundary condition value (°C for Dirichlet, W/m² for Neumann)

* **K** = Thermal Conductivity (W/m-K)

* **Lx** = Length in the X-direction (m)

* **Ly** = Length in the Y-direction (m)

* **Nx** = number of grid points in X-direction

* **Ny** = number of grid points in Y-direction

""")

# ============================================================
# USER INPUT
# ============================================================

st.sidebar.header("**Boundary**")

#Left
TL = st.sidebar.radio("**Left Boundary:**", ["Dirichlet", "Neumann"], index=0)
if TL == "Dirichlet":
    bc_left = st.sidebar.number_input("TL (°C):", value=500.0)
else:
    bc_left = st.sidebar.number_input("qL (W/m²):", value=0.0)
    
#Right
TR = st.sidebar.radio("**Right Boundary:**", ["Dirichlet", "Neumann"], index=1)
if TR == "Dirichlet":
    bc_right = st.sidebar.number_input("TR (°C):", value=100.0)
else:
    bc_right = st.sidebar.number_input("qR (W/m²):", value=0.0)
    
#Top
TT = st.sidebar.radio("**Top Boundary:**", ["Dirichlet", "Neumann"], index=1)
if TT == "Dirichlet":
    bc_top = st.sidebar.number_input("TT (°C):", value=200.0)
else:
    bc_top = st.sidebar.number_input("qT (W/m²):", value=0.0)

#Bottom
TB = st.sidebar.radio("**Bottom Boundary:**", ["Dirichlet", "Neumann"], index=0)
if TB == "Dirichlet":
    bc_bottom = st.sidebar.number_input("TB (°C):", value=25.0)
else:
    bc_bottom = st.sidebar.number_input("qB (W/m²):", value=0.0)

# PROPERTIES
st.sidebar.header("**Properties**")
k = st.sidebar.number_input("K (W/m-K):", value=50.000, format="%.3f")
Lx = st.sidebar.number_input("Lx (meter)", value=1.000, format="%.3f")
Ly = st.sidebar.number_input("Ly (meter)", value=0.500, format="%.3f")
Nx = st.sidebar.number_input("Nx (Grid Number-X)", value=61)
Ny = st.sidebar.number_input("Ny (Grid Number-Y)", value=41)

dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

# ---- Solver control ----
max_iter = 10000
tolerance = 1e-6

st.subheader("**Laplace Equation**")
st.latex(r'''\frac{\partial^2 T}{\partial x^2}+\frac{\partial^2 T}
         {\partial y^2}= 0''')

st.subheader("**Finite Discretization**")
st.latex(r'''\frac{T_{i+1,j} - 2T_{i,j} + T_{i-1,j}}{\Delta x^2} 
         + \frac{T_{i,j+1} - 2T_{i,j} + T_{i,j-1}}{\Delta y^2} = 0''')

st.subheader("**Gauss–Seidel Solver**")
st.latex(r'''
T_{i,j}^{(k+1)} =
\frac{
\left( T_{i+1,j}^{(k)} + T_{i-1,j}^{(k+1)} \right)\Delta y^2
+
\left( T_{i,j+1}^{(k)} + T_{i,j-1}^{(k+1)} \right)\Delta x^2
}{
2\left( \Delta x^2 + \Delta y^2 \right)
}
''')

st.subheader("**Dirichlet Boundary Condition**")
st.latex(r'''
T = T_{\mathrm{bc}}
''')
st.subheader("**Neumann Boundary Condition**")
st.latex(r'''
\frac{\partial T}{\partial n} = 0
\;\;\Rightarrow\;\;
T_{\text{boundary}} = T_{\text{adjacent}}
''')

st.subheader("**Convergence Criteria**")
st.latex(r'''
\max \left| T^{(k+1)} - T^{(k)} \right| < \varepsilon
''')

st.latex(r'''
\varepsilon = 10^{-6}
''')

# ============================================================
# INITIALIZATION
# ============================================================

T = np.zeros((Ny, Nx))
T_old = T.copy()

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Initial guess
T[:, :] = 75.0

# Apply Dirichlet boundaries initially
if TL == "Dirichlet":
    T[:, 0] = bc_left
if TR == "Dirichlet":   
    T[:, -1] = bc_right
if TT == "Dirichlet":
    T[-1, :] = bc_top
if TB == "Dirichlet":
    T[0, :] = bc_bottom
    
# ============================================================
# GAUSS–SEIDEL SOLVER (LAPLACE)
# ============================================================
st.subheader("**Result & Visualization**")

save_every = 1 # Save every n iterations (lagger number = faster simulation)
T_history = []
qmag_history = []
qx_history = []
qy_history = []

for it in range(max_iter):
    T_old[:, :] = T[:, :]

    for j in range(1, Ny - 1):
        for i in range(1, Nx - 1):
            T[j, i] = (
                (T[j, i+1] + T[j, i-1]) * dy**2 +
                (T[j+1, i] + T[j-1, i]) * dx**2
            ) / (2 * (dx**2 + dy**2))
            
    if it % save_every == 0:
        T_history.append(T.copy())
        
        # HEAT FLUX CALCULATION
        dTdy, dTdx = np.gradient(T, dy, dx)
        qx = -k * dTdx
        qy = -k * dTdy
        q_mag = np.sqrt(qx**2 + qy**2)

        qmag_history.append(q_mag.copy())
        qx_history.append(qx.copy())
        qy_history.append(qy.copy())
    
    # ---- Neumann BCs (insulated: dT/dn = 0) ----
    if TL == "Neumann":
        T[:, 0] = T[:, 1]

    if TR == "Neumann":
        T[:, -1] = T[:, -2]

    if TB == "Neumann":
        T[0, :] = T[1, :]

    if TT == "Neumann":
        T[-1, :] = T[-2, :]
    
    # ---- Dirichlet BCs (re-apply) ----
    if TL == "Dirichlet":
        T[:, 0] = bc_left

    if TR == "Dirichlet":
        T[:, -1] = bc_right
        
    if TB == "Dirichlet":
        T[0, :] = bc_bottom

    if TT == "Dirichlet":
        T[-1, :] = bc_top

    # ---- Convergence check ----
    error = np.max(np.abs(T - T_old))
    if error < tolerance:
        st.write(f"Converged in {it+1} iterations")
        break
    
T_history = np.array(T_history)
T_history = T_history.astype(np.float32)

qmag_history = np.array(qmag_history, dtype=np.float32)
qx_history = np.array(qx_history, dtype=np.float32)
qy_history = np.array(qy_history, dtype=np.float32)

n_iter = T_history.shape[0]
n_frame = qmag_history.shape[0]

# ============================================================
# VISUALIZATION
# ============================================================

# ============================================================
# 3D Visualization
# ============================================================
#Temperature Distribution Slice Plot
import plotly.graph_objects as go

slice_mode = st.radio("Temperature slice direction",
                      ["X-index (Vertical Slice)", 
                       "Y-index (Horizontal Slice)"]
)
if slice_mode == "X-index (Vertical Slice)":
    idx = st.slider("Select X-index", 0, Nx - 1, Nx // 2)

    T_slice = T_history[:, :, idx]   # shape: (n_iter, Ny)

    x_mesh = np.arange(Ny)           # spatial (y direction)
    y_mesh = np.arange(n_iter)       # iteration
    x_mesh, y_mesh = np.meshgrid(x_mesh, y_mesh)

    x_label = "Y-index"
else:
    idx = st.slider("Select Y-index", 0, Ny - 1, Ny // 2)

    T_slice = T_history[:, idx, :]   # shape: (n_iter, Nx)

    x_mesh = np.arange(Nx)           # spatial (x direction)
    y_mesh = np.arange(n_iter)       # iteration
    x_mesh, y_mesh = np.meshgrid(x_mesh, y_mesh)

    x_label = "X-index"

fig = go.Figure(
    data=[
        go.Surface(
            z=T_slice,
            x=x_mesh,
            y=y_mesh,
            colorscale="Magma"
        )
    ]
)

fig.update_layout(
    scene=dict(
        xaxis_title=x_label,
        yaxis_title="Iteration",
        zaxis_title="Temperature (°C)",
    ),
    height=700,
)

st.plotly_chart(fig, use_container_width=True)

#Heat Flux Distribution Slice Plot
qmag_history.shape == (n_iter, Ny, Nx)

slice_mode_q = st.radio("Heat Flux slice direction",
                      ["X-index (Vertical Slice)", 
                       "Y-index (Horizontal Slice)"])

if slice_mode_q == "X-index (Vertical Slice)":
    idx = st.slider(
        "Select X-index (Heat Flux)",
        0, Nx - 1, Nx // 2,
        key="qx_idx"
    )

    q_slice = qmag_history[:, :, idx]   # (n_iter, Ny)

    x_mesh = np.arange(Ny)
    y_mesh = np.arange(n_iter)
    x_mesh, y_mesh = np.meshgrid(x_mesh, y_mesh)

    x_label = "Y-index"
else:
    idx = st.slider(
        "Select Y-index (Heat Flux)",
        0, Ny - 1, Ny // 2,
        key="qy_idx"
    )

    q_slice = qmag_history[:, idx, :]   # (n_iter, Nx)

    x_mesh = np.arange(Nx)
    y_mesh = np.arange(n_iter)
    x_mesh, y_mesh = np.meshgrid(x_mesh, y_mesh)

    x_label = "X-index"


fig_q = go.Figure(
    data=[
        go.Surface(
            z=q_slice,
            x=x_mesh,
            y=y_mesh,
            colorscale="Viridis"
        )
    ]
)

fig_q.update_layout(
    scene=dict(
        xaxis_title=x_label,
        yaxis_title="Iteration",
        zaxis_title="Heat Flux (W/m²)"
    ),
    height=700,
)

st.plotly_chart(fig_q, use_container_width=True)

# ============================================================
#Autoplay Settings
# ============================================================
st.sidebar.header("**Autoplay Settings**")
button = st.sidebar.radio("Animation:", ["Temperature Distribution", 
                                         "Heat Flux Magnitude", 
                                         "Heat Flux Vector Field", 
                                         "Heat Flux Streamline", 
                                         "Heat Flux Direction & Magnitude"], index=None)
stop = st.sidebar.button("STOP Animation")
if stop:
    button = index=None
frame = st.sidebar.number_input("Frame Delay (ms)", 
                              min_value=1,
                              max_value=2000,
                              value=10)

choose = st.sidebar.radio("**Iteration Selection Mode:**", ["Slider", "Input Number"], index=0)

#Temperature Distribution (Heat Map)
plot_area = st.empty()

vmin = np.min(T_history)
vmax = np.max(T_history)

def plot_frame(k):
    fig, ax = plt.subplots(figsize=(7, 4))

    c = ax.imshow(
        T_history[k],
        extent=[0, Lx, 0, Ly],
        origin="lower",
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
        aspect="auto"
    )

    ax.set_title(f"Temperature Distribution (Iteration {k * save_every})")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    fig.colorbar(c, ax=ax, label="Temperature (°C)")

    plot_area.pyplot(fig)
    plt.close(fig)

# Heat Flux Magnitude
plot_area_q = st.empty()

n_frame = qmag_history.shape[0]

vmin_q = np.min(qmag_history)
vmax_q = np.max(qmag_history)

def plot_frame_q(n):
    fig, ax = plt.subplots(figsize=(7, 4))

    c = ax.contourf(
        X, Y,
        qmag_history[n],
        levels=30,
        cmap="viridis",
        vmin=vmin_q,
        vmax=vmax_q
    )

    ax.set_title(f"Heat Flux Magnitude (Iteration {n * save_every})")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    fig.colorbar(c, ax=ax, label="Heat Flux (W/m²)")

    plot_area_q.pyplot(fig)
    plt.close(fig)


if "hf_iter" not in st.session_state:
    st.session_state.hf_iter = 0

# Heat Flux Vector Field
st.subheader("Heat Flux Vector Field")

plot_area_vec = st.empty()

vmin_T = T_history.min()
vmax_T = T_history.max()

skip = 2  # biar panah nggak rame 

def plot_vector_frame(m):
    fig, ax = plt.subplots(figsize=(7, 4))

    cf = ax.contourf(
        X, Y,
        T_history[m],
        levels=20,
        cmap="inferno",
        vmin=vmin_T,
        vmax=vmax_T
    )

    ax.quiver(
        X[::skip, ::skip],
        Y[::skip, ::skip],
        qx_history[m][::skip, ::skip],
        qy_history[m][::skip, ::skip],
        color="cyan",
        scale=50000
    )

    ax.set_title(f"Heat Flux Vector Field (Iteration {m * save_every})")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    fig.colorbar(cf, ax=ax, label="Temperature (°C)")

    plot_area_vec.pyplot(fig)
    plt.close(fig)

# Heat Flux Streamlines
st.subheader("Heat Flux Streamlines")

plot_area_stream = st.empty()

vmin_q = qmag_history.min()
vmax_q = qmag_history.max()

def plot_stream_frame(h):
    fig, ax = plt.subplots(figsize=(7, 4))

    strm = ax.streamplot(
        X, Y,
        qx_history[h],
        qy_history[h],
        color=qmag_history[h],
        cmap="viridis",
        density=1.0,
        linewidth=1
    )

    ax.set_title(f"Heat Flux Streamlines (Iteration {h * save_every})")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    cbar = fig.colorbar(strm.lines, ax=ax)
    cbar.set_label("Heat Flux (W/m²)")

    plot_area_stream.pyplot(fig)
    plt.close(fig)

# Direction + Magnitude via Colormap
st.subheader("Heat Flux Direction & Magnitude")

plot_area_dir = st.empty()

skip = 3  # biar panah nggak rame

vmin_q = qmag_history.min()
vmax_q = qmag_history.max()

def plot_dir_mag_frame(n):
    qx_n = qx_history[n] / (qmag_history[n] + 1e-12)
    qy_n = qy_history[n] / (qmag_history[n] + 1e-12)

    fig, ax = plt.subplots(figsize=(7, 4))

    q = ax.quiver(
        X[::skip, ::skip],
        Y[::skip, ::skip],
        qx_n[::skip, ::skip],
        qy_n[::skip, ::skip],
        qmag_history[n][::skip, ::skip],
        cmap="viridis",
        scale=30
    )

    ax.set_title(f"Heat Flux Direction & Magnitude (Iteration {n * save_every})")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    cbar = fig.colorbar(q, ax=ax)
    cbar.set_label("Heat Flux (W/m²)")

    plot_area_dir.pyplot(fig)
    plt.close(fig)

# ============================================================
# RUN AUTOPLAY AND SELECTION MODE
# ============================================================
# Temperature Distribution
if button == "Temperature Distribution":
    for n in range(n_iter):
        plot_frame(n)
        time.sleep(frame / 1000)
else:
    if choose == "Slider":
        n = st.sidebar.slider("Temperature Distribution", 0, n_iter - 1, 0)
    else:
        n = st.sidebar.number_input("Temperature Distribution", min_value=0, max_value=n_iter - 1, value=0, step=1)
    plot_frame(n)

# Heat Flux Magnitude
if button == "Heat Flux Magnitude":
    for n in range(n_frame):
        st.session_state.hf_iter = n
        plot_frame_q(n)
        time.sleep(frame / 1000)
else:
    if choose == "Slider":
        st.session_state.hf_iter = st.sidebar.slider(
            "Heat Flux Magnitude",
            0,
            n_frame - 1,
            st.session_state.hf_iter,
            key="hf_slider"
        )
    else:
        st.session_state.hf_iter = st.sidebar.number_input(
            "Heat Flux Magnitude",
            min_value=0,
            max_value=n_frame - 1,
            value=st.session_state.hf_iter,
            step=1,
            key="hf_number"
        )
    plot_frame_q(st.session_state.hf_iter)

# Heat Flux Vector Field
if button == "Heat Flux Vector Field":
    for m in range(n_frame):
        plot_vector_frame(m)
        time.sleep(frame / 1000)
else:
    if choose == "Slider":
        m = st.sidebar.slider("Heat Flux Vector Field", 0, n_frame - 1, 0)
    else:
        m = st.sidebar.number_input("Heat Flux Vector Field", min_value=0, max_value=n_frame - 1, value=0, step=1)
    plot_vector_frame(m)

#Heat Flux Streamlines
if button == "Heat Flux Streamline":
    for h in range(n_frame):
        plot_stream_frame(h)
        time.sleep(frame / 1000)
else:
    if choose == "Slider":
        h = st.sidebar.slider("Heat Flux Streamline", 0, n_frame - 1, 0)
    else:
        h = st.sidebar.number_input("Heat Flux Streamline", min_value=0, max_value=n_frame - 1, value=0, step=1)
    plot_stream_frame(h)

# Direction + Magnitude via Colormap
if button == "Heat Flux Direction & Magnitude":
    for n in range(n_frame):
        plot_dir_mag_frame(n)
        time.sleep(frame / 1000)
else:
    if choose == "Slider":
        n = st.sidebar.slider("Heat Flux Direction & Magnitude", 0, n_frame - 1, 0)
    else:
        n = st.sidebar.number_input("Heat Flux Direction & Magnitude", min_value=0, max_value=n_frame - 1, value=0, step=1)
    plot_dir_mag_frame(n)

# Print Data Frame
import pandas as pd

# Temperature Data Frame
df = pd.DataFrame(T, columns=[f"x={xi:.2f}m" for xi in x], index=[f"y={yi:.2f}m" for yi in y])
st.subheader("**Temperature Data Frame**")
st.dataframe(df) 
# Heat Flux Data Frame
dfx = pd.DataFrame(qx, columns=[f"x={xi:.2f}m" for xi in x], index=[f"y={yi:.2f}m" for yi in y])
dfy = pd.DataFrame(qy, columns=[f"x={xi:.2f}m" for xi in x], index=[f"y={yi:.2f}m" for yi in y])
st.subheader("**Heat Flux Data Frame (X-direction)**")
st.dataframe(dfx)
st.subheader("**Heat Flux Data Frame (Y-direction)**")
st.dataframe(dfy)




