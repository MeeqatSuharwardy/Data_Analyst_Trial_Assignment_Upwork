import pandas as pd
import numpy as np

keys_df = pd.read_csv('KEYS.csv')
census_population_df = pd.read_csv('CENSUS_POPULATION_STATE.tsv', sep='\t')
census_mhi_df = pd.read_csv('CENSUS_MHI_STATE.csv')
redfin_price_df = pd.read_csv('REDFIN_MEDIAN_SALE_PRICE.csv')

filtered_keys = keys_df[(keys_df['region_type'] == 'state') & (~keys_df['key_row'].str.contains("'", na=False))].reset_index(drop=True)

output_df = pd.DataFrame()
output_df['key_row'] = filtered_keys['key_row']
output_df['StateCode'] = filtered_keys['zillow_region_name']

def ordinal_suffix(rank):
    if 11 <= rank % 100 <= 13:
        return 'th'
    return {1: 'st', 2: 'nd', 3: 'rd'}.get(rank % 10, 'th')

def get_population(code):
    row = census_population_df[census_population_df.iloc[:, 0].str.strip() == "Total population"]
    return row.get(f"{code}!!Estimate", pd.Series([np.nan])).values[0]

output_df['census_population'] = output_df['StateCode'].apply(get_population)

output_df['population_rank'] = output_df['census_population'].str.replace(',', '').astype(float).rank(ascending=False, method='min').astype(int)
output_df['population_blurb'] = output_df.apply(
    lambda row: f"{row['key_row']} is {row['population_rank']}{ordinal_suffix(row['population_rank'])} in the nation in population among states, DC, and Puerto Rico.",
    axis=1
)

def get_income(code):
    row = census_mhi_df[census_mhi_df.iloc[:, 0].str.strip() == "Households"]
    return row.get(f"{code}!!Median income (dollars)!!Estimate", pd.Series([np.nan])).values[0]

output_df['median_household_income'] = output_df['StateCode'].apply(get_income)
output_df['median_household_income'] = (
    output_df['median_household_income'].astype(str)
    .str.replace(r'[$,]', '', regex=True)
    .replace('nan', np.nan)
    .astype(float)
)

output_df['median_household_income_rank'] = output_df['median_household_income'].rank(ascending=False, method='min').astype('Int64')

def income_blurb(row):
    rank = row['median_household_income_rank']
    if pd.isna(rank):
        return ""
    if rank == 1:
        return f"{row['key_row']} is the highest in the nation in median household income among states, DC, and Puerto Rico."
    return f"{row['key_row']} is {rank}{ordinal_suffix(int(rank))} in the nation in median household income among states, DC, and Puerto Rico."

output_df['median_household_income_blurb'] = output_df.apply(income_blurb, axis=1)

latest_month_label = redfin_price_df.columns[-1]

def get_sale_price(code):
    row_index = redfin_price_df[redfin_price_df.iloc[:, 0] == code].index
    if row_index.empty:
        return np.nan
    val = redfin_price_df.loc[row_index[0], latest_month_label]
    if pd.isna(val):
        return np.nan
    return float(str(val).replace('K', '000').replace('$', '').replace(',', ''))

output_df['median_sale_price'] = output_df['StateCode'].apply(get_sale_price)
output_df['median_sale_price_rank'] = output_df['median_sale_price'].rank(ascending=False, method='min').astype('Int64')

def sale_price_blurb(row):
    rank = row['median_sale_price_rank']
    if pd.isna(rank):
        return ""
    if rank == 1:
        return f"{row['key_row']} has the single highest median sale price on homes in the nation among states, DC, and Puerto Rico, according to Redfin data from {latest_month_label}."
    return f"{row['key_row']} has the {rank}{ordinal_suffix(int(rank))} highest median sale price on homes in the nation among states, DC, and Puerto Rico, according to Redfin data from {latest_month_label}."

output_df['median_sale_price_blurb'] = output_df.apply(sale_price_blurb, axis=1)

output_df['house_affordability_ratio'] = (output_df['median_sale_price'] / output_df['median_household_income']).round(1)
output_df['house_affordability_ratio_rank'] = output_df['house_affordability_ratio'].rank(ascending=True, method='min').astype('Int64')

def affordability_blurb(row):
    rank = row['house_affordability_ratio_rank']
    if pd.isna(rank):
        return ""
    if rank == 1:
        return f"{row['key_row']} has the single lowest house affordability ratio in the nation among states, DC, and Puerto Rico, according to Redfin data from {latest_month_label}."
    return f"{row['key_row']} has the {rank}{ordinal_suffix(int(rank))} lowest house affordability ratio in the nation among states, DC, and Puerto Rico, according to Redfin data from {latest_month_label}."

output_df['house_affordability_ratio_blurb'] = output_df.apply(affordability_blurb, axis=1)

final_df = output_df[[
    'key_row',
    'census_population',
    'population_rank',
    'population_blurb',
    'median_household_income',
    'median_household_income_rank',
    'median_household_income_blurb',
    'median_sale_price',
    'median_sale_price_rank',
    'median_sale_price_blurb',
    'house_affordability_ratio',
    'house_affordability_ratio_rank',
    'house_affordability_ratio_blurb'
]]

final_df.to_csv('Analysis_of_States.csv', index=False)
print("Final summary saved as 'Analysis_of_States.csv'")
