# ============================================================
#  Route Optimization for Delivery Services — app.py
#  Run locally:  streamlit run app.py
#  Requirements: pip install streamlit pandas numpy scikit-learn
#                            xgboost networkx matplotlib seaborn
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.linear_model     import LinearRegression
from sklearn.ensemble         import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics          import mean_absolute_error, mean_squared_error, r2_score
from xgboost                  import XGBRegressor

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title = "Route Optimization — Delivery Services",
    page_icon  = "🚚",
    layout     = "wide",
)

# ─────────────────────────────────────────────
#  CITY COORDINATES
# ─────────────────────────────────────────────
CITY_COORDS = {
    'Warehouse'    : (20.5937, 78.9629),
    'Mumbai'       : (19.0760, 72.8777),
    'Delhi'        : (28.6139, 77.2090),
    'Bangalore'    : (12.9716, 77.5946),
    'Chennai'      : (13.0827, 80.2707),
    'Kolkata'      : (22.5726, 88.3639),
    'Hyderabad'    : (17.3850, 78.4867),
    'Pune'         : (18.5204, 73.8567),
    'Ahmedabad'    : (23.0225, 72.5714),
    'Jaipur'       : (26.9124, 75.7873),
    'Surat'        : (21.1702, 72.8311),
    'Lucknow'      : (26.8467, 80.9462),
    'Nagpur'       : (21.1458, 79.0882),
    'Indore'       : (22.7196, 75.8577),
    'Bhopal'       : (23.2599, 77.4126),
    'Visakhapatnam': (17.6868, 83.2185),
    'Patna'        : (25.5941, 85.1376),
    'Vadodara'     : (22.3072, 73.1812),
    'Coimbatore'   : (11.0168, 76.9558),
    'Kochi'        : ( 9.9312, 76.2673),
}

# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R    = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a    = (np.sin(dlat/2)**2 +
            np.cos(np.radians(lat1)) *
            np.cos(np.radians(lat2)) *
            np.sin(dlon/2)**2)
    return R * 2 * np.arcsin(np.sqrt(a))


@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)


@st.cache_data
def preprocess(_df):
    df_ml = _df.copy()
    le_v  = LabelEncoder()
    le_t  = LabelEncoder()
    le_p  = LabelEncoder()
    le_d  = LabelEncoder()
    df_ml['vehicle_enc']  = le_v.fit_transform(df_ml['vehicle_type'])
    df_ml['traffic_enc']  = le_t.fit_transform(df_ml['traffic_condition'])
    df_ml['priority_enc'] = le_p.fit_transform(df_ml['priority'])
    df_ml['dest_enc']     = le_d.fit_transform(df_ml['destination'])
    df_ml['cost_per_km']  = (df_ml['fuel_cost_inr'] /
                              df_ml['distance_km']).round(2)
    df_ml['time_per_km']  = (df_ml['travel_time_hrs'] /
                              df_ml['distance_km']).round(4)
    df_ml['efficiency']   = (df_ml['num_packages'] /
                              (df_ml['travel_time_hrs'] *
                               df_ml['fuel_cost_inr'] + 1)).round(4)
    return df_ml


@st.cache_resource
def train_models(_df_ml):
    FEATURES = ['distance_km','vehicle_enc','traffic_enc',
                'priority_enc','num_packages','effective_speed',
                'cost_per_km','time_per_km']
    TARGET   = 'travel_time_hrs'
    X        = _df_ml[FEATURES]
    y        = _df_ml[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    scaler    = StandardScaler()
    Xtr       = scaler.fit_transform(X_train)
    Xte       = scaler.transform(X_test)

    models = {
        'Linear Regression' : LinearRegression(),
        'Random Forest'     : RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting' : GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost'           : XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    }
    results = {}
    for name, mdl in models.items():
        mdl.fit(Xtr, y_train)
        yp  = mdl.predict(Xte)
        results[name] = {
            'model' : mdl,
            'mae'   : mean_absolute_error(y_test, yp),
            'rmse'  : np.sqrt(mean_squared_error(y_test, yp)),
            'r2'    : r2_score(y_test, yp),
            'y_pred': yp,
            'y_test': y_test,
        }
    best = max(results, key=lambda k: results[k]['r2'])
    return results, best, scaler, FEATURES


@st.cache_resource
def build_graph(_df):
    G   = nx.Graph()
    for city, (lat, lon) in CITY_COORDS.items():
        G.add_node(city, lat=lat, lon=lon)
    summary = (_df.groupby(['origin','destination'])
                  .agg(avg_time =('travel_time_hrs','mean'),
                       avg_dist =('distance_km','mean'),
                       avg_cost =('fuel_cost_inr','mean'))
                  .reset_index())
    for _, row in summary.iterrows():
        G.add_edge(row['origin'], row['destination'],
                   weight   = round(row['avg_time'], 3),
                   distance = round(row['avg_dist'], 1),
                   cost     = round(row['avg_cost'], 1))
    return G


def dijkstra_route(G, src, dst):
    try:
        path   = nx.dijkstra_path(G, src, dst, weight='weight')
        length = nx.dijkstra_path_length(G, src, dst, weight='weight')
        dist   = sum(G[path[i]][path[i+1]].get('distance', 0)
                     for i in range(len(path)-1))
        cost   = sum(G[path[i]][path[i+1]].get('cost', 0)
                     for i in range(len(path)-1))
        return path, round(length, 2), round(dist, 1), round(cost, 1)
    except Exception:
        return None, None, None, None


def nn_tsp(matrix, cities, start=0):
    n         = len(cities)
    unvisited = set(range(n)) - {start}
    route, cur, total = [start], start, 0
    while unvisited:
        nxt    = min(unvisited, key=lambda j: matrix[cur][j])
        total += matrix[cur][nxt]
        route.append(nxt); unvisited.remove(nxt); cur = nxt
    total += matrix[cur][start]; route.append(start)
    return route, round(total, 2)


def two_opt(matrix, route):
    best      = route[:]
    best_cost = sum(matrix[best[i]][best[i+1]] for i in range(len(best)-1))
    improved  = True
    while improved:
        improved = False
        for i in range(1, len(best)-2):
            for j in range(i+1, len(best)-1):
                new  = best[:i] + best[i:j+1][::-1] + best[j+1:]
                cost = sum(matrix[new[k]][new[k+1]] for k in range(len(new)-1))
                if cost < best_cost:
                    best, best_cost, improved = new, cost, True
    return best, round(best_cost, 2)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/delivery-truck.png", width=72)
st.sidebar.title("🚚 Route Optimization")
st.sidebar.markdown("**Delivery Services — Data Science Project**")
st.sidebar.markdown("---")

uploaded = st.sidebar.file_uploader(
    "📂 Upload delivery_routes.csv", type=["csv"])

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview",
     "📊 EDA",
     "🤖 ML Models",
     "🗺️ Route Finder",
     "🔄 TSP Optimizer",
     "🔮 Predict Trip Time"],
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Dataset:** 300 routes · 19 cities\n\n"
    "**Algorithms:** Dijkstra · NN-TSP · 2-Opt\n\n"
    "**Models:** LR · RF · GBM · XGBoost"
)

# ─────────────────────────────────────────────
#  GUARD — need data
# ─────────────────────────────────────────────
if uploaded is None:
    st.title("🚚 Route Optimization for Delivery Services")
    st.info("👈  Upload **delivery_routes.csv** from the sidebar to begin.")
    st.markdown("""
    ### What this app does
    | Section | Description |
    |---------|-------------|
    | 🏠 Overview | Dataset summary & key metrics |
    | 📊 EDA | Charts exploring distances, costs, traffic |
    | 🤖 ML Models | Train & compare 4 regression models |
    | 🗺️ Route Finder | Dijkstra shortest path between any two cities |
    | 🔄 TSP Optimizer | Visit all cities with minimum travel time |
    | 🔮 Predict Trip Time | Enter route details and get a time estimate |
    """)
    st.stop()

# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
df     = load_data(uploaded)
df_ml  = preprocess(df)
G      = build_graph(df)
results, best_ml, scaler, FEATURES = train_models(df_ml)

# ─────────────────────────────────────────────
#  PAGE: OVERVIEW
# ─────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("🏠 Project Overview")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Routes",    len(df))
    c2.metric("Cities Covered",  df['destination'].nunique())
    c3.metric("Avg Distance",    f"{df['distance_km'].mean():.1f} km")
    c4.metric("Avg Travel Time", f"{df['travel_time_hrs'].mean():.2f} hrs")
    c5.metric("Success Rate",    f"{df['delivery_success'].mean()*100:.1f}%")

    st.markdown("---")
    st.subheader("📋 Raw Dataset")
    st.dataframe(df, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Column Info")
        info_df = pd.DataFrame({
            'Column' : df.columns,
            'Type'   : df.dtypes.values,
            'Non-Null': df.notnull().sum().values,
            'Sample' : df.iloc[0].values,
        })
        st.dataframe(info_df, use_container_width=True)
    with col2:
        st.subheader("Numerical Summary")
        st.dataframe(df.describe().round(2), use_container_width=True)


# ─────────────────────────────────────────────
#  PAGE: EDA
# ─────────────────────────────────────────────
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    palette = {'Truck':'#3498db', 'Van':'#2ecc71', 'Bike':'#e74c3c'}

    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    fig.patch.set_facecolor('#f8f9fa')
    fig.suptitle('Delivery Routes — EDA', fontsize=14, fontweight='bold')

    # 1
    ax  = axes[0,0]
    vc  = df['vehicle_type'].value_counts()
    bars = ax.bar(vc.index, vc.values,
                  color=[palette[v] for v in vc.index],
                  edgecolor='white', width=0.5)
    for b,v in zip(bars,vc.values):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
                str(v), ha='center', fontweight='bold')
    ax.set_title('Vehicle Distribution', fontweight='bold')
    ax.set_facecolor('#f8f9fa'); ax.spines[['top','right']].set_visible(False)
    ax.set_ylim(0, max(vc.values)*1.2)

    # 2
    ax  = axes[0,1]
    tc  = df['traffic_condition'].value_counts()
    ax.pie(tc.values, labels=tc.index,
           colors=['#27ae60','#f39c12','#e74c3c'],
           autopct='%1.1f%%', startangle=90,
           wedgeprops={'edgecolor':'white'})
    ax.set_title('Traffic Conditions', fontweight='bold')

    # 3
    ax = axes[0,2]
    ax.hist(df['distance_km'], bins=30,
            color='#3498db', edgecolor='white', alpha=0.85)
    ax.axvline(df['distance_km'].mean(), color='#e74c3c',
               linestyle='--', lw=2,
               label=f"Mean: {df['distance_km'].mean():.0f} km")
    ax.set_title('Distance Distribution', fontweight='bold')
    ax.set_xlabel('Distance (km)'); ax.legend(fontsize=9)
    ax.set_facecolor('#f8f9fa'); ax.spines[['top','right']].set_visible(False)

    # 4
    ax = axes[1,0]
    grp = (df.groupby(['vehicle_type','traffic_condition'])
             ['travel_time_hrs'].mean().unstack()
             .reindex(index=['Bike','Van','Truck'],
                      columns=['Low','Medium','High']))
    grp.plot(kind='bar', ax=ax,
             color=['#27ae60','#f39c12','#e74c3c'],
             edgecolor='white', width=0.7, rot=0)
    ax.set_title('Avg Travel Time — Vehicle vs Traffic', fontweight='bold')
    ax.set_ylabel('Avg Travel Time (hrs)')
    ax.legend(title='Traffic', fontsize=8)
    ax.set_facecolor('#f8f9fa'); ax.spines[['top','right']].set_visible(False)

    # 5
    ax = axes[1,1]
    for vt, col in palette.items():
        ax.hist(df[df['vehicle_type']==vt]['fuel_cost_inr'],
                bins=20, alpha=0.6, color=col, label=vt, density=True)
    ax.set_title('Fuel Cost by Vehicle', fontweight='bold')
    ax.set_xlabel('Fuel Cost (₹)'); ax.legend(fontsize=9)
    ax.set_facecolor('#f8f9fa'); ax.spines[['top','right']].set_visible(False)

    # 6
    ax = axes[1,2]
    for vt, col in palette.items():
        m = df['vehicle_type']==vt
        ax.scatter(df.loc[m,'distance_km'], df.loc[m,'travel_time_hrs'],
                   color=col, alpha=0.5, s=25, label=vt)
    ax.set_title('Distance vs Travel Time', fontweight='bold')
    ax.set_xlabel('Distance (km)'); ax.set_ylabel('Travel Time (hrs)')
    ax.legend(fontsize=9)
    ax.set_facecolor('#f8f9fa'); ax.spines[['top','right']].set_visible(False)

    # 7
    ax    = axes[2,0]
    top10 = df['destination'].value_counts().head(10)
    ax.barh(top10.index[::-1], top10.values[::-1],
            color='#9b59b6', edgecolor='white')
    ax.set_title('Top 10 Destinations', fontweight='bold')
    ax.set_xlabel('Route Count')
    ax.set_facecolor('#f8f9fa'); ax.spines[['top','right']].set_visible(False)

    # 8
    ax = axes[2,1]
    pr = df['priority'].value_counts()
    ax.bar(pr.index, pr.values,
           color=['#95a5a6','#f39c12','#e74c3c'],
           edgecolor='white', width=0.5)
    ax.set_title('Priority Distribution', fontweight='bold')
    ax.set_facecolor('#f8f9fa'); ax.spines[['top','right']].set_visible(False)

    # 9
    ax      = axes[2,2]
    num_c   = ['distance_km','travel_time_hrs','fuel_cost_inr',
                'effective_speed','num_packages']
    corr    = df[num_c].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, ax=ax, linewidths=0.5,
                annot_kws={'size':8}, square=True)
    ax.set_title('Feature Correlation', fontweight='bold')
    ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    st.pyplot(fig)


# ─────────────────────────────────────────────
#  PAGE: ML MODELS
# ─────────────────────────────────────────────
elif page == "🤖 ML Models":
    st.title("🤖 ML Models — Travel Time Prediction")

    st.subheader("Model Performance")
    perf_data = []
    for name, res in results.items():
        perf_data.append({
            'Model'    : name,
            'MAE'      : round(res['mae'],  4),
            'RMSE'     : round(res['rmse'], 4),
            'R² Score' : round(res['r2'],   4),
        })
    perf_df = pd.DataFrame(perf_data).sort_values('R² Score', ascending=False)
    st.dataframe(perf_df.style.highlight_max(
        subset=['R² Score'], color='#d4edda')
        .highlight_min(subset=['MAE','RMSE'], color='#d4edda'),
        use_container_width=True)
    st.success(f"🏆 Best model: **{best_ml}**  "
               f"(R² = {results[best_ml]['r2']:.4f})")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        names   = list(results.keys())
        r2s     = [results[m]['r2'] for m in names]
        cols_bar= ['#e74c3c' if m==best_ml else '#3498db' for m in names]
        bars    = ax.bar(names, r2s, color=cols_bar, edgecolor='white', width=0.5)
        for b,v in zip(bars,r2s):
            ax.text(b.get_x()+b.get_width()/2,
                    b.get_height()+0.005,
                    f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')
        ax.set_title('R² Score Comparison', fontweight='bold')
        ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_facecolor('#f8f9fa')
        ax.spines[['top','right']].set_visible(False)
        fig.patch.set_facecolor('#f8f9fa')
        st.pyplot(fig)

    with col2:
        fig, ax   = plt.subplots(figsize=(7, 4))
        y_pred_b  = results[best_ml]['y_pred']
        y_test_b  = results[best_ml]['y_test']
        ax.scatter(y_test_b, y_pred_b,
                   alpha=0.5, color='#3498db', s=30)
        mn = min(y_test_b.min(), y_pred_b.min())
        mx = max(y_test_b.max(), y_pred_b.max())
        ax.plot([mn,mx],[mn,mx],'r--', lw=2, label='Perfect')
        ax.set_xlabel('Actual (hrs)'); ax.set_ylabel('Predicted (hrs)')
        ax.set_title(f'Actual vs Predicted — {best_ml}', fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_facecolor('#f8f9fa')
        ax.spines[['top','right']].set_visible(False)
        fig.patch.set_facecolor('#f8f9fa')
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Feature Importance (Random Forest)")
    rf_imp  = results['Random Forest']['model'].feature_importances_
    fi_df   = pd.Series(rf_imp, index=FEATURES).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    fi_df.plot(kind='barh', ax=ax, color='#3498db', edgecolor='white')
    ax.set_title('Feature Importance', fontweight='bold')
    ax.set_xlabel('Importance')
    ax.set_facecolor('#f8f9fa')
    ax.spines[['top','right']].set_visible(False)
    fig.patch.set_facecolor('#f8f9fa')
    st.pyplot(fig)


# ─────────────────────────────────────────────
#  PAGE: ROUTE FINDER
# ─────────────────────────────────────────────
elif page == "🗺️ Route Finder":
    st.title("🗺️ Shortest Route Finder — Dijkstra's Algorithm")

    cities_all = list(CITY_COORDS.keys())
    col1, col2 = st.columns(2)
    with col1:
        src = st.selectbox("📍 Origin",      cities_all, index=0)
    with col2:
        dst = st.selectbox("🎯 Destination", cities_all, index=1)

    if st.button("🔍 Find Shortest Route", use_container_width=True):
        if src == dst:
            st.warning("Origin and destination are the same!")
        else:
            path, time, dist, cost = dijkstra_route(G, src, dst)
            if path:
                st.success(f"✅ Optimal route found — "
                           f"{len(path)-1} hop(s)")
                c1, c2, c3 = st.columns(3)
                c1.metric("⏱ Travel Time", f"{time:.2f} hrs")
                c2.metric("📏 Distance",   f"{dist:.1f} km")
                c3.metric("⛽ Fuel Cost",  f"₹{cost:.0f}")

                st.markdown("**Route:**  " +
                            "  →  ".join([f"**{p}**" for p in path]))

                # Draw graph with route highlighted
                pos = {city:(lon,lat)
                       for city,(lat,lon) in CITY_COORDS.items()
                       if city in G.nodes()}
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.set_facecolor('#eaf3fb')
                fig.patch.set_facecolor('#eaf3fb')
                nx.draw_networkx_edges(
                    G, pos, ax=ax, edge_color='#cccccc',
                    width=0.6, alpha=0.4)
                route_edges = [(path[i], path[i+1])
                               for i in range(len(path)-1)
                               if G.has_edge(path[i], path[i+1])]
                nx.draw_networkx_edges(
                    G, pos, edgelist=route_edges, ax=ax,
                    edge_color='#e74c3c', width=3, alpha=0.9)
                nc = ['#e74c3c' if n == src
                      else '#27ae60' if n == dst
                      else '#2ecc71' if n in path
                      else '#aaaaaa' for n in G.nodes()]
                nx.draw_networkx_nodes(
                    G, pos, ax=ax, node_color=nc,
                    node_size=250, alpha=0.95)
                nx.draw_networkx_labels(
                    G, pos, ax=ax, font_size=7,
                    font_color='white', font_weight='bold')
                ax.set_title(f"Shortest Route: {src} → {dst}",
                             fontweight='bold')
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.error("No path found between selected cities.")


# ─────────────────────────────────────────────
#  PAGE: TSP OPTIMIZER
# ─────────────────────────────────────────────
elif page == "🔄 TSP Optimizer":
    st.title("🔄 TSP Route Optimizer — Nearest Neighbor + 2-Opt")

    st.info("Select the cities you want to include in the delivery tour.")
    cities_sel = st.multiselect(
        "Select delivery cities (Warehouse is always included)",
        options=[c for c in CITY_COORDS.keys() if c != 'Warehouse'],
        default=['Mumbai','Delhi','Bangalore','Chennai','Hyderabad','Pune'],
    )

    if len(cities_sel) < 2:
        st.warning("Select at least 2 cities.")
        st.stop()

    all_cities = ['Warehouse'] + cities_sel

    if st.button("🚀 Optimize Route", use_container_width=True):
        with st.spinner("Building time matrix and running optimization..."):
            n      = len(all_cities)
            matrix = np.full((n,n), 9999.0)
            np.fill_diagonal(matrix, 0)
            for i, c1 in enumerate(all_cities):
                for j, c2 in enumerate(all_cities):
                    if i != j:
                        try:
                            t = nx.dijkstra_path_length(
                                G, c1, c2, weight='weight')
                            matrix[i][j] = t
                        except Exception:
                            pass

            nn_r,  nn_c  = nn_tsp(matrix, all_cities, start=0)
            opt_r, opt_c = two_opt(matrix, nn_r)
            saving       = nn_c - opt_c
            pct          = saving/nn_c*100 if nn_c > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("NN Route Time",   f"{nn_c:.2f} hrs")
        c2.metric("2-Opt Route Time",f"{opt_c:.2f} hrs")
        c3.metric("Time Saved",      f"{saving:.2f} hrs")
        c4.metric("Improvement",     f"{pct:.1f}%")

        st.markdown("**Optimized route:**  " +
                    "  →  ".join([f"**{all_cities[i]}**" for i in opt_r]))

        # Comparison bar
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.patch.set_facecolor('#f8f9fa')

        ax = axes[0]
        bars = ax.bar(['Nearest\nNeighbor','2-Opt\nImproved'],
                      [nn_c, opt_c],
                      color=['#f39c12','#27ae60'],
                      edgecolor='white', width=0.45)
        for b,v in zip(bars,[nn_c,opt_c]):
            ax.text(b.get_x()+b.get_width()/2,
                    b.get_height()+0.3,
                    f'{v:.2f} hrs', ha='center', fontweight='bold')
        ax.set_title('Algorithm Comparison', fontweight='bold')
        ax.set_ylabel('Total Time (hrs)')
        ax.set_facecolor('#f8f9fa')
        ax.spines[['top','right']].set_visible(False)
        ax.set_ylim(0, max(nn_c, opt_c)*1.2)

        ax  = axes[1]
        pos = {city:(lon,lat)
               for city,(lat,lon) in CITY_COORDS.items()
               if city in all_cities and city in G.nodes()}
        sub = G.subgraph([c for c in all_cities if c in G.nodes()])
        nx.draw_networkx_edges(sub, pos, ax=ax,
                               edge_color='#cccccc', width=0.8, alpha=0.4)
        opt_edges = []
        for k in range(len(opt_r)-1):
            c1n = all_cities[opt_r[k]]
            c2n = all_cities[opt_r[k+1]]
            if G.has_edge(c1n, c2n):
                opt_edges.append((c1n, c2n))
        nx.draw_networkx_edges(sub, pos, edgelist=opt_edges, ax=ax,
                               edge_color='#e74c3c', width=2.5, alpha=0.9)
        nc = ['#e74c3c' if n=='Warehouse' else '#3498db'
              for n in sub.nodes()]
        nx.draw_networkx_nodes(sub, pos, ax=ax,
                               node_color=nc, node_size=280, alpha=0.95)
        nx.draw_networkx_labels(sub, pos, ax=ax,
                                font_size=7, font_color='white',
                                font_weight='bold')
        ax.set_title('Optimized Delivery Tour', fontweight='bold')
        ax.set_facecolor('#eaf3fb')
        ax.axis('off')

        plt.tight_layout()
        st.pyplot(fig)


# ─────────────────────────────────────────────
#  PAGE: PREDICT TRIP TIME
# ─────────────────────────────────────────────
elif page == "🔮 Predict Trip Time":
    st.title("🔮 Predict Travel Time for a New Route")
    st.markdown(f"Using **{best_ml}** (best model, "
                f"R² = {results[best_ml]['r2']:.4f})")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        dist      = st.slider("Distance (km)",       50,  2500, 900)
        packages  = st.slider("Number of packages",   1,    25,  10)
    with col2:
        vehicle   = st.selectbox("Vehicle type",
                                  ['Truck','Van','Bike'])
        traffic   = st.selectbox("Traffic condition",
                                  ['Low','Medium','High'])
    with col3:
        priority  = st.selectbox("Priority level",
                                  ['Low','Medium','High'])
        speed_map = {'Truck':60,'Van':70,'Bike':40}
        tmult     = {'Low':1.0,'Medium':0.8,'High':0.6}
        eff_spd   = speed_map[vehicle] * tmult[traffic]
        st.metric("Effective Speed", f"{eff_spd:.0f} km/h")

    if st.button("🔮 Predict Travel Time", use_container_width=True):
        fuel_rate  = {'Truck':0.35,'Van':0.25,'Bike':0.10}
        fuel_cost  = dist * fuel_rate[vehicle] * 95
        cost_per_km= fuel_cost / dist
        time_per_km= (dist / eff_spd) / dist

        le_v = LabelEncoder().fit(['Bike','Truck','Van'])
        le_t = LabelEncoder().fit(['High','Low','Medium'])
        le_p = LabelEncoder().fit(['High','Low','Medium'])
        le_d = LabelEncoder().fit(df['destination'].unique())

        sample = pd.DataFrame([{
            'distance_km'  : dist,
            'vehicle_enc'  : le_v.transform([vehicle])[0],
            'traffic_enc'  : le_t.transform([traffic])[0],
            'priority_enc' : le_p.transform([priority])[0],
            'num_packages' : packages,
            'effective_speed': eff_spd,
            'cost_per_km'  : round(cost_per_km, 2),
            'time_per_km'  : round(time_per_km, 4),
        }])

        sample_s = scaler.transform(sample[FEATURES])
        pred_time= results[best_ml]['model'].predict(sample_s)[0]

        st.success(f"### ⏱ Predicted Travel Time: **{pred_time:.2f} hours**")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Distance",     f"{dist} km")
        c2.metric("Vehicle",       vehicle)
        c3.metric("Fuel Cost",    f"₹{fuel_cost:,.0f}")
        c4.metric("Eff. Speed",   f"{eff_spd:.0f} km/h")

        # Sensitivity: time vs distance
        dists    = np.linspace(50, 2500, 60)
        preds    = []
        for d in dists:
            fc   = d * fuel_rate[vehicle] * 95
            cpk  = fc / d
            tpk  = (d / eff_spd) / d
            row  = pd.DataFrame([{
                'distance_km'   : d,
                'vehicle_enc'   : le_v.transform([vehicle])[0],
                'traffic_enc'   : le_t.transform([traffic])[0],
                'priority_enc'  : le_p.transform([priority])[0],
                'num_packages'  : packages,
                'effective_speed': eff_spd,
                'cost_per_km'   : round(cpk, 2),
                'time_per_km'   : round(tpk, 4),
            }])
            preds.append(results[best_ml]['model']
                         .predict(scaler.transform(row[FEATURES]))[0])

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(dists, preds, color='#3498db', lw=2.5)
        ax.axvline(dist, color='#e74c3c', linestyle='--', lw=2,
                   label=f'Your route ({dist} km)')
        ax.axhline(pred_time, color='#e74c3c', linestyle=':', lw=1.5)
        ax.scatter([dist], [pred_time], color='#e74c3c', s=80, zorder=5)
        ax.set_xlabel('Distance (km)', fontsize=11)
        ax.set_ylabel('Predicted Time (hrs)', fontsize=11)
        ax.set_title('Predicted Travel Time vs Distance', fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_facecolor('#f8f9fa')
        ax.spines[['top','right']].set_visible(False)
        fig.patch.set_facecolor('#f8f9fa')
        st.pyplot(fig)
