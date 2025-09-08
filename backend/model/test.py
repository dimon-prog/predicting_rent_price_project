import asyncio
import httpx
import pandas as pd
import json
from typing import List, Dict, Any

BASE_URL = "https://www.immobilienscout24.at/portal/graphql"


def parse_graphql_data(json_response: Dict[str, Any]) -> List[Dict[str, Any]]:

    parsed_items = []

    try:
        hits = json_response['data']['findPropertiesByParams']['hits']
    except (KeyError, TypeError):
        print("Не удалось найти 'hits' в ответе. Ответ:")
        print(json_response)
        return []
    print(hits)
    for item in hits:
        parsed_items.append({
            'id': item.get('exposeId'),
            'headline': item.get('headline'),
            'address': item.get('addressString'),
            'price_eur': item.get('primaryPrice'),
            'area_sqm': item.get('primaryArea'),
            'rooms': item.get('numberOfRooms'),
            'company': item.get('realtorContact', {}).get('company'),
            'url': "https://www.immobilienscout24.at" + item.get('links', {}).get('targetURL', ''),
        })
    return parsed_items


async def main():
    all_results = []
    total_pages_to_scrape = 1  # Сколько страниц парсим
    results_per_page = 20  # Сколько объявлений на странице (проверьте в URL или на сайте)

    # Статические части нашего запроса, они не меняются
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

            # Динамически формируем 'variables' для каждой страницы
            variables = {
                "aspectRatio": 1.77,
                "params": {
                    "countryCode": "AT",
                    "estateType": "APARTMENT",
                    "from": str((page_num - 1) * results_per_page),  # Рассчитываем смещение
                    "region": "009001",  # Код региона для Вены, похоже
                    "size": str(results_per_page),  # Количество результатов
                    "transferType": "RENT",
                    "useType": "RESIDENTIAL"
                }
            }

            # Собираем все параметры для GET-запроса
            # Важно: словари variables и extensions нужно преобразовать в строку JSON
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
                    break  # Прерываем цикл, если данных больше нет

                all_results.extend(parsed)
                print(f" -> Добавлено {len(parsed)} объявлений.")
                await asyncio.sleep(0.5)  # Небольшая пауза из вежливости

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
    df.to_csv('immoscout_graphql_listings.csv', index=False, encoding='utf-8-sig')

    print("\nDataFrame успешно создан и сохранен в 'immoscout_graphql_listings.csv'")
    print(df.head())


if __name__ == "__main__":
    asyncio.run(main())