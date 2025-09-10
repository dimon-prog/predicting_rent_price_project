import asyncio
import httpx
import pandas as pd
import json
from typing import List, Dict, Any
import re

BASE_URL = "https://www.immobilienscout24.at/portal/graphql"


def parse_graphql_data(json_response: Dict[str, Any]) -> List[Dict[str, Any]]:

    parsed_items = []


    hits = json_response['data']['findPropertiesByParams']['hits']
    print(hits)
    for item in hits:
        badges = item.get('badges', [])
        address = item.get('addressString', '')
        badge_values = {badge['value'] for badge in badges if 'value' in badge}
        badge_labels = {badge['label'].lower() for badge in badges if 'label' in badge and badge['label']}
        parsed_items.append({
            'id': item.get('exposeId'),
            'address': address,
            'price': item.get('primaryPrice'),
            'area_sqm': item.get('primaryArea'),
            'rooms': item.get('numberOfRooms'),
            'is_new_building': 1 if 'neubauprojekt' in badge_labels else 0,
            "has_balcony": 1 if 'BALCONY' in badge_values else 0,
            "no_commission": 1 if 'FREE_OF_COMMISSION' in badge_values else 0,
            "has_terrace": 1 if 'TERRACE' in badge_values else 0,
            "is_furnished": 1 if 'FURNISHED_FULL' in badge_values else 0,
            "is_social_housing": 1 if item.get('isSocialHousing') else 0,
            "district": re.search(r"\d{4}", item.get("addressString")).group(0)})

    return parsed_items


async def main():
    all_results = []
    total_pages_to_scrape = 200
    results_per_page = 20


    operation_name = "findPropertiesByParams"
    extensions = {
        "persistedQuery": {
            "version": 1,
            "sha256Hash": "87eb384abe72fbd1289b5a394281ea07132f24c7396ebe66456511303f3fcf8b"
        }
    }

    async with httpx.AsyncClient() as client:
        for page_num in range(1, total_pages_to_scrape + 1):
            print(f"Запрашиваем страницу {page_num}...")


            variables = {
                "aspectRatio": 1.77,
                "params": {
                    "countryCode": "AT",
                    "estateType": "APARTMENT",
                    "from": str((page_num - 1) * results_per_page),
                    "region": "009001",
                    "size": str(results_per_page),
                    "transferType": "RENT",
                    "useType": "RESIDENTIAL"
                }
            }


            params = {
                "operationName": operation_name,
                "variables": json.dumps(variables, separators=(',', ':')),
                "extensions": json.dumps(extensions, separators=(',', ':'))
            }

            try:
                response = await client.get(BASE_URL, params=params, timeout=20.0)
                response.raise_for_status()
                data = response.json()

                parsed = parse_graphql_data(data)
                if not parsed:
                    print(f"На странице {page_num} не найдено объявлений. Возможно, это последняя страница.")
                    break

                all_results.extend(parsed)
                print(f" -> Добавлено {len(parsed)} объявлений.")
                await asyncio.sleep(0.5)

            except httpx.RequestError as e:
                print(f"Ошибка при запросе страницы {page_num}: {e}")
                break
            except Exception as e:
                print(f"Непредвиденная ошибка на странице {page_num}: {e}")
                break

    if not all_results:
        print("\nНе удалось собрать ни одного объявления.")
        return

    print(f"\nВсего собрано {len(all_results)} объявлений.")

    df = pd.DataFrame(all_results)
    df.to_csv('vienna_apartments.csv', index=False, encoding='utf-8-sig')

    print("\nDataFrame успешно создан и сохранен в 'vienna_apartments.csv'")
    print(df.head())


if __name__ == "__main__":
    asyncio.run(main())