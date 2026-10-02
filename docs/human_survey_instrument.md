# Human survey instrument

Each evaluator viewed the same 200 generated product images in a randomized
order and answered the following fields for every image.

1. Is the image recognizable as a product? (`yes`, `partial`, `no`)
2. What is the overall image quality? (1--5)
3. How recognizable is the target product category? (1--5)
4. How confident are you in this judgment? (1--5)
5. Which design intent is most prominent? (`functionality`, `aesthetics`,
   `usability`, `symbolism`, `unclear`)
6. Is the image more function-oriented or aesthetic-oriented? (1--5; 1 is
   strongly function-oriented, 3 is balanced, and 5 is strongly
   aesthetic-oriented)
7. How well does the visible form communicate the product function? (1--5)
8. If the image or judgment is ambiguous, briefly state why. (optional text)

The operational machine-readable version is
`code/schema/human_survey_schema.json`. The exact VLM prompt uses the same
shared fields plus five VLM-only interpretive fields and is stored in
`code/schema/s2_8plus5.py`.

