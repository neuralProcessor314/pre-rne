# Plate layout shared by segment.py, train.py and infer.py (the same in every experiment)

# Condition codes: DD = DMSO + DMSO, DF = DMSO + ferrostatin, RF = RSL3 + ferrostatin, RD = RSL3 + DMSO
CODES = ["RD", "RF", "DD", "DF"]

# Condition of each plate row, top to bottom (wells 1-6, 7-12, 13-18, 19-24)
ROW_CODES = ["RD", "RF", "DD", "DF"]

# Plate columns from left to right: (side, concentration)
# The 1.0 columns (wells 5, 6, 11, 12, 17, 18, 23, 24) stay in the list so well numbers keep their plate positions
COLUMNS = [("L", "0.1"), ("R", "0.1"), ("L", "0.5"), ("R", "0.5"), ("L", "1.0"), ("R", "1.0")]
CONCENTRATIONS = ["0.1", "0.5"]     # 1.0 dropped: every cell was already dead


# Well number 1-24 of a condition, side and concentration, e.g. ("DD", "L", "0.1") -> 13
def well_number(code, side, conc):
    row = ROW_CODES.index(code)
    column = COLUMNS.index((side, conc))
    return row * 6 + column + 1


# Condition code, side and concentration of a well number, e.g. 13 -> ("DD", "L", "0.1")
def well_position(number):
    row = (number - 1) // 6
    column = (number - 1) % 6
    side, conc = COLUMNS[column]
    return ROW_CODES[row], side, conc


# Well name of a well number, e.g. 13 -> "DD_L0.1"
def well_name(number):
    code, side, conc = well_position(number)
    return code + "_" + side + conc
