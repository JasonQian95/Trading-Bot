signal_name = "Signal"
default_signal = ""
buy_signal = "Buy"
sell_signal = "Sell"
soft_buy_signal = "SoftBuy"
soft_sell_signal = "SoftSell"
signals = [buy_signal, sell_signal, soft_buy_signal, soft_sell_signal]
signal_colors = {
    buy_signal: "green",
    sell_signal: "red",
    soft_buy_signal: "yellow",
    soft_sell_signal: "yellow"
}
signal_markers = {
    buy_signal: "^",
    sell_signal: "v",
    soft_buy_signal: "^",
    soft_sell_signal: "v"
}


class InsufficientDataException(Exception):
    pass
