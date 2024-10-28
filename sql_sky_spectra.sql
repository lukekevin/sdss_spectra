SELECT TOP 1000
s.ObjType, s.fiberID, s.specObjID,s.plateID, s.mjd,
s.ra, s.dec

FROM SpecPhotoAll as s

WHERE s.ObjType='SKY' AND s.ra BETWEEN 145 AND 150 AND s.dec BETWEEN 25 AND 30 


