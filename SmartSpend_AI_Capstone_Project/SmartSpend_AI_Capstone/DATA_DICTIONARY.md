# Data Dictionary

| Field | Type | Description |
|---|---|---|
| `transaction_id` | String | Synthetic identifier; not a real account or bank reference |
| `date` | Date-like string | Synthetic transaction date in YYYY-MM-DD format |
| `description` | String | Human-readable transaction description used as model input |
| `amount` | Float | Synthetic transaction amount in Indian rupees |
| `category` | String | Ground-truth category label used for supervised learning |
| `data_source` | String | Indicates that the row is synthetic |

## Categories
Food, Groceries, Transportation, Shopping, Utilities, Healthcare, Education, Entertainment, Personal Care, Housing.

## Data limitations
The dataset is handcrafted for demonstration, is small, and may not represent real transaction language or spending distributions. It should not be used to make claims about real-world banking accuracy.
