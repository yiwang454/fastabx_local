import json
ACC2SPSK_TEST = {
   "SouthernEnglish": [
       "p268", "p273", "p274", "p276"
   ],
   "NorthernEnglish": [
       "p277", "p282", "p286", "p287",
   ],
   "Scottish": [
       "p264", "p265", "p284", "p285"
   ],
   "Irish": [
       "p298", "p313", "p340", "p364"
   ],
   "NorthernIrish": [
       "p292", "p293", "p304", "p351"
   ],
   "SouthAfrican": [
       "p314", "p323", "p336", "p347"
   ], # 1M3F
   "Indian": [
       "p248", "p251", "p376"
   ], # 2M1F
   "Oceanian": [
       "p326", "p335", "p374"
   ], # 2M1F
   "Canadian": [
       "p316", "p317", "p343", "p363"
   ],
   "AmericanNortheast":[
       "p315", "p339", "p360", "p361"
   ],
   "AmericanMidwest":[
       "p311", "p333", "p334", "p341"
   ],
   "AmericanSouth":[
       "p301", "p308", "p310", "p345"
   ], # 1M3F
   "AmericanWest":[
       "p294", "p299", "p300", "p318"
   ], # 4F
}

with open("/home/s2522559/datastore/items_vctk/split.json", "w+") as file_w:
    json.dump(ACC2SPSK_TEST, file_w)
