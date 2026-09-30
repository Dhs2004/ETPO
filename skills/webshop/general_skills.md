# WebShop initial skills

## When to use
Use for shopping tasks that require finding and purchasing a product matching a natural-language request.

## Workflow
1. Extract the requested product type, required attributes, options, and maximum price from the task. Keep these constraints visible while searching.
2. Search using the product type and a few discriminative attributes. Refine the query when results are irrelevant.
3. Open candidate product pages and inspect their description, features, price, and available options. Product titles alone may omit required constraints.
4. Select the requested size, color, quantity, or other options before buying. Use the exact clickable labels shown by the environment.
5. Check the product and selected options against every requested constraint before using the buy action.

## Pitfalls
Do not invent product identifiers or option labels. A product with the right category can still fail because of an incorrect attribute, option, or price. Searching and viewing pages do not complete a purchase. Follow the environment's admissible search[...] and click[...] actions and stop when the environment terminates.
