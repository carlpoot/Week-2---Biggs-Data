from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
OUTPUT_DIR = BASE_DIR / 'outputs'
for folder in [RAW_DIR, PROCESSED_DIR, OUTPUT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

LISTINGS_URL = "https://data.insideairbnb.com/united-states/ma/boston/2025-12-27/data/listings.csv.gz"
CALENDAR_URL = "https://data.insideairbnb.com/united-states/ma/boston/2025-12-27/data/calendar.csv.gz"
REVIEWS_URL = "https://data.insideairbnb.com/united-states/ma/boston/2025-12-27/data/reviews.csv.gz"

print('Downloading real data from Inside Airbnb...')
listings = pd.read_csv(LISTINGS_URL, compression='gzip', low_memory=False)
calendar = pd.read_csv(CALENDAR_URL, compression='gzip', low_memory=False)
reviews = pd.read_csv(REVIEWS_URL, compression='gzip', low_memory=False)

listings.to_csv(RAW_DIR / 'boston_listings_raw.csv', index=False)
calendar.to_csv(RAW_DIR / 'boston_calendar_raw.csv', index=False)
reviews.to_csv(RAW_DIR / 'boston_reviews_raw.csv', index=False)

print('Raw row counts:')
print('listings:', len(listings))
print('calendar:', len(calendar))
print('reviews:', len(reviews))
print('total:', len(listings) + len(calendar) + len(reviews))

listing_cols = [
    'id','name','host_id','host_is_superhost','neighbourhood_cleansed','latitude','longitude',
    'property_type','room_type','accommodates','bathrooms_text','bedrooms','beds','amenities',
    'price','minimum_nights','maximum_nights','availability_365','number_of_reviews',
    'review_scores_rating','instant_bookable'
]
listings_clean = listings[listing_cols].copy()
listings_clean['price'] = listings_clean['price'].replace('[\$,]', '', regex=True).astype(float)
for col in ['bedrooms','beds','review_scores_rating']:
    listings_clean[col] = pd.to_numeric(listings_clean[col], errors='coerce')
listings_clean['host_is_superhost'] = listings_clean['host_is_superhost'].fillna('f')
listings_clean['instant_bookable'] = listings_clean['instant_bookable'].fillna('f')
listings_clean['bathrooms_text'] = listings_clean['bathrooms_text'].fillna('Unknown')
listings_clean['amenities_count'] = listings_clean['amenities'].fillna('').str.count(',') + 1
listings_clean['price_band'] = pd.cut(
    listings_clean['price'],
    bins=[0,100,200,350,10000],
    labels=['Budget','Mid-range','Upper-mid','Premium'],
    include_lowest=True
)
listings_clean = listings_clean.drop_duplicates(subset=['id']).reset_index(drop=True)

calendar_clean = calendar[['listing_id','date','available','price','adjusted_price','minimum_nights','maximum_nights']].copy()
calendar_clean['date'] = pd.to_datetime(calendar_clean['date'], errors='coerce')
for col in ['price','adjusted_price']:
    calendar_clean[col] = calendar_clean[col].replace('[\$,]', '', regex=True)
    calendar_clean[col] = pd.to_numeric(calendar_clean[col], errors='coerce')
calendar_clean['available_flag'] = calendar_clean['available'].map({'t':1, 'f':0})
calendar_clean['year_month'] = calendar_clean['date'].dt.to_period('M').astype(str)
calendar_clean = calendar_clean.dropna(subset=['listing_id','date']).drop_duplicates().reset_index(drop=True)

review_cols = ['listing_id','id','date','reviewer_id','reviewer_name','comments']
reviews_clean = reviews[review_cols].copy()
reviews_clean['date'] = pd.to_datetime(reviews_clean['date'], errors='coerce')
reviews_clean['comment_length'] = reviews_clean['comments'].fillna('').str.len()
reviews_clean = reviews_clean.dropna(subset=['listing_id','id','date']).drop_duplicates(subset=['id']).reset_index(drop=True)

review_summary = reviews_clean.groupby('listing_id').agg(
    total_reviews=('id','count'),
    avg_comment_length=('comment_length','mean'),
    latest_review=('date','max')
).reset_index()

calendar_summary = calendar_clean.groupby('listing_id').agg(
    avg_calendar_price=('price','mean'),
    availability_rate=('available_flag','mean'),
    booked_days=('available_flag', lambda s: int((1 - s).sum()))
).reset_index()

analytics = listings_clean.merge(review_summary, how='left', left_on='id', right_on='listing_id')
analytics = analytics.merge(calendar_summary, how='left', left_on='id', right_on='listing_id', suffixes=('','_calendar'))
analytics = analytics.drop(columns=['listing_id','listing_id_calendar'], errors='ignore')

listings_clean.to_csv(PROCESSED_DIR / 'boston_listings_clean.csv', index=False)
calendar_clean.to_csv(PROCESSED_DIR / 'boston_calendar_clean.csv', index=False)
reviews_clean.to_csv(PROCESSED_DIR / 'boston_reviews_clean.csv', index=False)
analytics.to_csv(PROCESSED_DIR / 'boston_boardie_analytics.csv', index=False)

plt.figure(figsize=(8,5))
listings_clean['room_type'].value_counts().plot(kind='bar')
plt.title('Boston Listings by Room Type')
plt.xlabel('Room Type')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'chart_room_type.png', dpi=150)
plt.close()

plt.figure(figsize=(8,5))
listings_clean['price'].dropna().clip(upper=listings_clean['price'].quantile(0.95)).hist(bins=30)
plt.title('Boston Listing Price Distribution (clipped at 95th percentile)')
plt.xlabel('Price per Night (USD)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'chart_price_distribution.png', dpi=150)
plt.close()

monthly = calendar_clean.groupby('year_month')['available_flag'].mean()
plt.figure(figsize=(10,5))
monthly.plot()
plt.title('Average Calendar Availability by Month')
plt.xlabel('Year-Month')
plt.ylabel('Availability Rate')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'chart_monthly_availability.png', dpi=150)
plt.close()

by_neigh = listings_clean.groupby('neighbourhood_cleansed')['price'].median().sort_values(ascending=False).head(15)
plt.figure(figsize=(10,6))
by_neigh.plot(kind='bar')
plt.title('Top 15 Boston Areas by Median Listing Price')
plt.xlabel('Neighbourhood')
plt.ylabel('Median Price (USD)')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'chart_neighbourhood_prices.png', dpi=150)
plt.close()

print('
Saved files:')
for path in sorted(BASE_DIR.rglob('*')):
    if path.is_file():
        print(path.relative_to(BASE_DIR))
