import numpy as np
import pandas as pd
import os
import random
import sklearn
from sklearn.model_selection import train_test_split
import rasterio
import xarray as xr  


# Set seed using random number generator once, save manually for reuse every time 
#SEED = int(os.environ.get("GLOBAL_SEED",random.SystemRandom().randint(0,2**32-1)))

SEED = 399786328

# Load the .npz file directly (no extraction needed)
z_tr = np.load(r"species_train_with_climate.npz", allow_pickle=True)
z_te = np.load(r"species_test_with_climate.npz", allow_pickle=True)


#CREATING TRAINING DATA
# Extract training data
train_locs   = z_tr["train_locs"]            # (N, 2) [lat, lon]
train_ids    = z_tr["train_ids"]             # (N,)
taxon_ids_tr = z_tr["taxon_ids"]             # (500,)

#Getting Training rows
N = train_locs.shape[0]

# Base train DF
df_train = pd.DataFrame({
    "latitude":  train_locs[:, 0],
    "longitude": train_locs[:, 1],
    "species_id": train_ids,
})

# Add per-row feature arrays (same length as train_locs)
skip_tr = {"train_locs", "train_ids", "taxon_ids"}
for k in z_tr.files:
    if k in skip_tr:
        continue
    arr = z_tr[k]
    if isinstance(arr, np.ndarray) and arr.shape == (N,):
        df_train[k] = arr
    elif isinstance(arr, np.ndarray) and arr.shape == (N, 1):
        df_train[k] = arr[:, 0]


#CREATING TEST DATA
test_locs     = z_te["test_locs"]            # (M, 2)
# z_te['test_pos_inds'] is a list of lists, where each list corresponds to 
# the indices in test_locs where a given species is present, it can be assumed 
# that they are not present in the other locations 
# test_pos_inds corresponds to the number of species after checking the shape of it.
# so number of species is equal to the taxon_ids.
test_pos_inds = dict(zip(z_te['taxon_ids'], z_te['test_pos_inds']))    


# Getting test rows
M = test_locs.shape[0]
print(M)
# Base test DF
df_test = pd.DataFrame({
    "latitude":  test_locs[:, 0],
    "longitude": test_locs[:, 1],
})

# Add per-row feature arrays (same length as test_locs)
skip_te = {"test_locs", "taxon_ids", "test_pos_inds"}
for k in z_te.files:
    if k in skip_te:
        continue
    arr = z_te[k]
    if isinstance(arr, np.ndarray) and arr.shape == (M,):
        df_test[k] = arr
    elif isinstance(arr, np.ndarray) and arr.shape == (M, 1):
        df_test[k] = arr[:, 0]

# print("df_test shape: \n", df_test.head())
# This is just dropping the percipitation columns, remove the line below, once the percipitation data is removed from the .npz (the old data)
df_test = df_test.iloc[:,:-3]

# Bioclim feature addition
# Directory with your WorldClim/Bioclim tifs
BIOCLIM_DIR = r"D:\AML\wc"

# Bioclim precipitation variable names
precip_codes = [f"wc2.1_10m_bio_{n}" for n in range(12, 20)]  # bio12..bio19
solar_codes = [f"wc2.1_10m_bio_{n}" for n in range(12, 20)]  # bio12..bio19
wind_codes = [f"wc2.1_10m_bio_{n}" for n in range(12, 20)]  # bio12..bio19
vapour_codes = [f"wc2.1_10m_bio_{n}" for n in range(12, 20)]  # bio12..bio19

def find_precip_rasters(base_dir=BIOCLIM_DIR, codes=precip_codes):
    """Return dict: { 'bio12': 'path/to/bio12.tif', ... }"""
    rasters = {}
    for fname in os.listdir(base_dir):
        lower = fname.lower()
        for code in codes:
            if code in lower and fname.lower().endswith(".tif"):
                rasters[code] = os.path.join(base_dir, fname)
    return rasters

precip_rasters = find_precip_rasters()
print("Found precipitation rasters:")
for k, v in precip_rasters.items():
    print(k, "->", v)

def add_precip_to_df(df, precip_rasters, lat_col="latitude", lon_col="longitude"):
    """
    For each bio12..bio19 raster, sample at df's lat/lon and
    add as a new column (same name as key, e.g. 'bio12').
    """
    # (x, y) = (lon, lat) for rasterio
    coords = list(zip(df[lon_col].values, df[lat_col].values))

    for code, path in precip_rasters.items():
        print(f"Sampling {code} from {path}")
        with rasterio.open(path) as src:
            # sample() yields arrays of shape (1,) per point
            vals = np.array([v[0] for v in src.sample(coords)])
            # Convert nodata to NaN if needed
            if src.nodata is not None:
                vals = np.where(vals == src.nodata, np.nan, vals)
        df[code] = vals

    return df

# Add precipitation data to training and test dataframes
df_train = add_precip_to_df(df_train, precip_rasters)
df_test  = add_precip_to_df(df_test,  precip_rasters)

# Additional environmental variables
elev_folder = r"D:\AML\elevation"
srad_folder = r"D:\AML\solar"
wind_folder = r"D:\AML\wind"
vapr_folder = r"D:\AML\watervapour"

def find_rasters(base_dir, pattern):
    """Return dict: { 'elevation': 'path/to/elevation.tif', ... }"""
    rasters = {}
    for fname in os.listdir(base_dir):
        if pattern in fname.lower() and fname.lower().endswith(".tif"):
            rasters[fname.split('_')[-1]] = os.path.join(base_dir, fname)
    return rasters

def add_elevation_to_df(df, elev_folder, lat_col="latitude", lon_col="longitude"):
    """Sample wc2.1_10m_elev.tif at df's lat/lon and add as 'elevation'."""
    elev_path = os.path.join(elev_folder, "wc2.1_10m_elev.tif")
    if not os.path.exists(elev_path):
        raise FileNotFoundError(f"Elevation raster not found: {elev_path}")

    coords = list(zip(df[lon_col].values, df[lat_col].values))

    print(f"Sampling elevation from {elev_path}")
    with rasterio.open(elev_path) as src:
        vals = np.array([v[0] for v in src.sample(coords)])
        if src.nodata is not None:
            vals = np.where(vals == src.nodata, np.nan, vals)
    df["elevation"] = vals
    return df


def add_monthly_to_df(df, folder, var_name, lat_col="latitude", lon_col="longitude"):
    """
    Sample monthly rasters wc2.1_10m_{var_name}_01.tif ... _12.tif
    at df's lat/lon and add columns var_name_01..var_name_12.
    """
    coords = list(zip(df[lon_col].values, df[lat_col].values))

    for i in range(1, 13):
        fname = f"wc2.1_10m_{var_name}_{i:02d}.tif"
        path = os.path.join(folder, fname)
        col_name = f"{var_name}_{i:02d}"

        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping {col_name}")
            continue

        print(f"Sampling {col_name} from {path}")
        with rasterio.open(path) as src:
            vals = np.array([v[0] for v in src.sample(coords)])
            if src.nodata is not None:
                vals = np.where(vals == src.nodata, np.nan, vals)
        df[col_name] = vals

    return df

# Add elevation data to training and test dataframes
df_train = add_elevation_to_df(df_train, elev_folder)
df_test = add_elevation_to_df(df_test, elev_folder)

# Add monthly solar radiation data to training and test dataframes
df_train = add_monthly_to_df(df_train, srad_folder, "srad")
df_test = add_monthly_to_df(df_test, srad_folder, "srad")

# Add monthly wind data to training and test dataframes
df_train = add_monthly_to_df(df_train, wind_folder, "wind")
df_test = add_monthly_to_df(df_test, wind_folder, "wind")

# Add monthly water vapour data to training and test dataframes
df_train = add_monthly_to_df(df_train, vapr_folder, "vapr")
df_test = add_monthly_to_df(df_test, vapr_folder, "vapr")

month_cols_srad = [f"srad_{m:02d}" for m in range(1, 13)]
month_cols_wind = [f"wind_{m:02d}" for m in range(1, 13)]
month_cols_vapr = [f"vapr_{m:02d}" for m in range(1, 13)]

# Derive annual summaries for solar radiation
df_train["srad_mean"] = df_train[month_cols_srad].mean(axis=1)
df_train["srad_min"]  = df_train[month_cols_srad].min(axis=1)
df_train["srad_max"]  = df_train[month_cols_srad].max(axis=1)
df_train["srad_std"]  = df_train[month_cols_srad].std(axis=1)

df_test["srad_mean"] = df_test[month_cols_srad].mean(axis=1)
df_test["srad_min"]  = df_test[month_cols_srad].min(axis=1)
df_test["srad_max"]  = df_test[month_cols_srad].max(axis=1)
df_test["srad_std"]  = df_test[month_cols_srad].std(axis=1)

# Derive annual summaries for wind
df_train["wind_mean"] = df_train[month_cols_wind].mean(axis=1)
df_train["wind_min"]  = df_train[month_cols_wind].min(axis=1)
df_train["wind_max"]  = df_train[month_cols_wind].max(axis=1)
df_train["wind_std"]  = df_train[month_cols_wind].std(axis=1)

df_test["wind_mean"] = df_test[month_cols_wind].mean(axis=1)
df_test["wind_min"]  = df_test[month_cols_wind].min(axis=1)
df_test["wind_max"]  = df_test[month_cols_wind].max(axis=1)
df_test["wind_std"]  = df_test[month_cols_wind].std(axis=1)

# Derive annual summaries for water vapour
df_train["vapr_mean"] = df_train[month_cols_vapr].mean(axis=1)
df_train["vapr_min"]  = df_train[month_cols_vapr].min(axis=1)
df_train["vapr_max"]  = df_train[month_cols_vapr].max(axis=1)
df_train["vapr_std"]  = df_train[month_cols_vapr].std(axis=1)

df_test["vapr_mean"] = df_test[month_cols_vapr].mean(axis=1)
df_test["vapr_min"]  = df_test[month_cols_vapr].min(axis=1)
df_test["vapr_max"]  = df_test[month_cols_vapr].max(axis=1)
df_test["vapr_std"]  = df_test[month_cols_vapr].std(axis=1)


# Marine Data Features
BIO_ORACLE_DIR = r"D:\AML\marineData"  

def add_biooracle_var(df, nc_path, var_name, out_col,
                      lat_col="latitude", lon_col="longitude"):
    """
    Sample a Bio-ORACLE NetCDF variable at df's lat/lon and add as `out_col`.
    Adjust `var_name` if the dataset uses a different variable name.
    """
    ds = xr.open_dataset(nc_path)

    if var_name not in ds.data_vars:
        print(f"Warning: var_name '{var_name}' not in {nc_path}. Available: {list(ds.data_vars)}")
        da = list(ds.data_vars.values())[0]  # fall back to first var
    else:
        da = ds[var_name]

    if "time" in da.dims:
        da = da.mean("time")

    lat_dim = "lat"  if "lat"  in da.dims else "latitude"
    lon_dim = "lon"  if "lon"  in da.dims else "longitude"

    lats = xr.DataArray(df[lat_col].values, dims="points")
    lons = xr.DataArray(df[lon_col].values, dims="points")

    da_sampled = da.sel({lat_dim: lats, lon_dim: lons}, method="nearest")
    df[out_col] = da_sampled.values
    return df

# Paths to your Bio-ORACLE files 
thetao_nc  = os.path.join(BIO_ORACLE_DIR, "thetao_baseline_2000_2019_depthsurf_4e3e_1426_a71d_U1762833159565.nc")
so_nc      = os.path.join(BIO_ORACLE_DIR, "so_baseline_2000_2019_depthsurf_a5c8_8d8a_48fb_U1762833163681.nc")
sws_nc     = os.path.join(BIO_ORACLE_DIR, "sws_baseline_2000_2019_depthsurf_43ae_926f_54a6_U1762833166926.nc")
chl_nc     = os.path.join(BIO_ORACLE_DIR, "chl_baseline_2000_2018_depthsurf_5fb9_cfee_a6ce_U1762833175420.nc")
phyc_nc    = os.path.join(BIO_ORACLE_DIR, "phyc_baseline_2000_2020_depthsurf_7d39_02af_cdbd_U1762833170152.nc")
terrain_nc = os.path.join(BIO_ORACLE_DIR, "terrain_characteristics_6d78_4f59_9e1f_U1762833180170.nc")

# Train
df_train = add_biooracle_var(df_train, thetao_nc,  "thetao_mean",      "bo_temp_mean")
df_train = add_biooracle_var(df_train, so_nc,      "so_mean",          "bo_sal_mean")
df_train = add_biooracle_var(df_train, sws_nc,     "sws_mean",         "bo_curr_mean")
df_train = add_biooracle_var(df_train, chl_nc,     "chl_mean",         "bo_chl_mean")
df_train = add_biooracle_var(df_train, phyc_nc,    "phyc_mean",        "bo_pp_mean")
df_train = add_biooracle_var(df_train, terrain_nc, "bathymetry_mean",  "bo_depth")

# Test
df_test = add_biooracle_var(df_test, thetao_nc,  "thetao_mean",     "bo_temp_mean")
df_test = add_biooracle_var(df_test, so_nc,      "so_mean",         "bo_sal_mean")
df_test = add_biooracle_var(df_test, sws_nc,     "sws_mean",        "bo_curr_mean")
df_test = add_biooracle_var(df_test, chl_nc,     "chl_mean",        "bo_chl_mean")
df_test = add_biooracle_var(df_test, phyc_nc,    "phyc_mean",       "bo_pp_mean")
df_test = add_biooracle_var(df_test, terrain_nc, "bathymetry_mean", "bo_depth")

marine_feat_cols = [
    "bo_temp_mean", "bo_sal_mean", "bo_curr_mean",
    "bo_chl_mean", "bo_pp_mean", "bo_depth",
]


#Data Encoding for Longitude and Latitude function
def encode_coords(df):
    df['lon_sin'] = np.sin(np.pi * df['longitude'])
    df['lon_cos'] = np.cos(np.pi * df['longitude'])
    df['lat_sin'] = np.sin(np.pi * df['latitude'])
    df['lat_cos'] = np.cos(np.pi * df['latitude'])
    return df

# Apply encoding to train dataframe
df_train = encode_coords(df_train)

# Add marine_flag column to indicate if location is marine (land_mask == 0)
df_train['marine_flag'] = (df_train['land_mask'] == 0).astype(int)
df_test['marine_flag'] = (df_test['land_mask'] == 0).astype(int)

# Create Feature columns
precip_feat_cols = [f"wc2.1_10m_bio_{n}" for n in range(12, 20)]
elev_cols = [col for col in df_train.columns if 'elev' in col.lower()]
feat_cols = ['temp_mean_1991_2024', 'temp_std_1991_2024',
             'land_mask', 'lon_sin', 'lon_cos', 'lat_sin', 'lat_cos', 'marine_flag'
            ] + precip_feat_cols + elev_cols + [
             'srad_mean', 'srad_min', 'srad_max', 'srad_std',
             'wind_mean', 'wind_min', 'wind_max', 'wind_std',
             'vapr_mean', 'vapr_min', 'vapr_max', 'vapr_std'
            ]+ marine_feat_cols


#Grouping by site_id for multilabeling
df_train["site_id"] = df_train.groupby(["lon_sin","lon_cos", "lat_sin", "lat_cos"], sort=False).ngroup()





# Build X: one row per site with the feature values
X_train = (df_train.groupby("site_id")[feat_cols].first()
       .sort_index())

# Build Y: site × species 0/1 matrix
df_train["present"] = 1
species_cols = np.sort(df_train["species_id"].unique())
y_train = (df_train.pivot_table(index="site_id", columns="species_id",
                    values="present", aggfunc="max", fill_value=0)
       .reindex(columns=species_cols, fill_value=0)
       .sort_index()
       .astype("uint8"))

# Combine X and Y into one big DF
train_df = pd.concat([X_train, y_train], axis=1)


# # Split the training data into train and validation
# # Splitting the training df into train and validation
# species_train,species_val = sklearn.model_selection.train_test_split(train_df,test_size=0.2,random_state = SEED)

# species_train.to_csv("species_train.tsv",sep="\t",encoding="utf-8",index=False)
# species_val.to_csv("species_val.tsv",sep="\t",encoding="utf-8",index=False)



#FOR TEST DATA
# Build Y: site × species 0/1 matrix
# Specify all species present, same as the given code for sp = np.random.choice(species)
species_cols = z_te["taxon_ids"]  # same order as in z_te

# Create a dataframe with the co-ords present
y = pd.DataFrame(test_locs, columns=["longitude", "latitude"])

# Loop over the species in the test and extract their location they are present in
for sp in species_cols:
    # Set the species as 0 for all locations first, this will also add the species ID as a column
    y[sp] = 0 
    # From the given code test_inds_pos is the locations where the selected species is present
    test_inds_pos = test_pos_inds[sp]
    # Locate in the dataframe created above where the species is present and assign at that column a value of 1
    y.loc[test_inds_pos, sp] = 1 
    # # This is just to confirm for an example "sp 17050" to make sure the dataframe is capturing the location correctly
    # if sp == 17050:
    #     pres = test_locs[test_inds_pos, 1], test_locs[test_inds_pos, 0]
    #     print("Lon",pres[0])
    #     print("Lat",pres[1])
    #     print(len(pres[0]))

        
# Test with the same sp 17050 from above to compare
# sp = 17050
# print(y.loc[y[sp] == 1, ["longitude", "latitude", sp]])
#print(len(y))



# Select the columns, drop the lon and lat, this is because the co-ords are in the same order
y = y[species_cols]

# Join the df_test with the dataframe built above
df_test = pd.concat([df_test, y], axis=1)

# # Confirm that it looks sensible
# print(df_test.head())

# Now apply encoding to test dataframe
df_test = encode_coords(df_test)


#Grouping by site_id for multilabeling
df_test["site_id"] = df_test.groupby(
    ["lon_sin", "lon_cos", "lat_sin", "lat_cos"], sort=False
).ngroup()

# Build X same as above
X_test = (df_test.groupby("site_id")[feat_cols].first()
       .sort_index())

# Build Y, as we have already done major part of it above
Y_test = (
    df_test.groupby("site_id")[species_cols]
           .max()          # 1 if species present at any row for that site
           .sort_index()
)

# Combine X and Y into one big DF
test_df = pd.concat([X_test, Y_test], axis=1)

# # Check with above that the data persisted and that there arent any errors, check the num of rows
# sp = 17050
# print(test_df.loc[test_df[sp] == 1, ['temp_mean_1991_2024', 'temp_std_1991_2024', 'land_mask', 'lon_sin', 'lon_cos', 'lat_sin', 'lat_cos',sp]])

print(train_df.head())
print(test_df.head())

# Save train_df and test_df using pickle (more reliable for DataFrames)
import pickle

with open('train_df_with_bioclim.pkl', 'wb') as f:
    pickle.dump(train_df, f)
with open('test_df_with_bioclim.pkl', 'wb') as f:
    pickle.dump(test_df, f)

print("\n✓ Saved train_df to 'train_df_with_bioclim.pkl'")
print("✓ Saved test_df to 'test_df_with_bioclim.pkl'")
