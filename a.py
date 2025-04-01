import deeplake
ds = deeplake.open("hub://activeloop/pacs-train")

print(ds.head())