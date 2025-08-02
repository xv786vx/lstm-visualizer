import React, { useEffect } from "react";
import "./App.css";

function App() {
  const [selectedTickers, setSelectedTickers] = React.useState([]);
  const [startDate, setStartDate] = React.useState("");
  const [endDate, setEndDate] = React.useState("");
  const [sequenceLength, setSequenceLength] = React.useState(30);
  const [plotUrls, setPlotUrls] = React.useState([]);
  const [metrics, setMetrics] = React.useState({});
  const [loading, setLoading] = React.useState(null);
  const [error, setError] = React.useState(null);

  const REACT_APP_API_URL = "https://lstm-visualizer-backend.onrender.com"; // for deployment
  // const REACT_APP_API_URL = "http://localhost:4999"; // for local testing

  // Popular stock tickers for selection
  const popularTickers = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "NVDA",
    "TSLA",
    "META",
    "NFLX",
    "JPM",
    "JNJ",
    "V",
    "PG",
    "UNH",
    "HD",
    "MA",
    "DIS",
    "PYPL",
    "ADBE",
    "CRM",
    "INTC",
    "VZ",
    "CMCSA",
    "PFE",
    "ABT",
    "KO",
    "PEP",
    "TMO",
    "COST",
  ];

  useEffect(() => {
    // Set default dates: 2020-01-01 to 2025-01-01
    setStartDate("2020-01-01");
    setEndDate("2025-01-01");
  }, []);

  const handleTickerToggle = (ticker) => {
    setSelectedTickers((prev) => {
      if (prev.includes(ticker)) {
        return prev.filter((t) => t !== ticker);
      } else {
        if (prev.length >= 8) {
          setError("Maximum 8 stocks allowed");
          setTimeout(() => setError(null), 3000);
          return prev;
        }
        return [...prev, ticker];
      }
    });
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (selectedTickers.length === 0) {
      setError("Please select at least one stock");
      return;
    }

    setLoading(true);
    setPlotUrls([]);
    setMetrics({});
    setError(null);

    console.log("submitting", {
      tickers: selectedTickers,
      start_date: startDate,
      end_date: endDate,
      seq_length: sequenceLength,
    });

    const response = await fetch(`${REACT_APP_API_URL}/train`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        tickers: selectedTickers,
        start_date: startDate,
        end_date: endDate,
        seq_length: sequenceLength,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      setError(
        errorData.error || "Error generating prediction. Please check inputs"
      );
      setLoading(false);
      return;
    }

    const eventSource = new EventSource(`${REACT_APP_API_URL}/train/stream`);

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.progress) {
        setLoading(data.progress);
      } else if (data.plot_urls && data.metrics) {
        setPlotUrls(data.plot_urls);
        setMetrics(data.metrics);
        setLoading(false);
        eventSource.close();
      }
    };

    eventSource.onerror = () => {
      setError("Error generating prediction. Please check inputs");
      setLoading(false);
      eventSource.close();
    };
  };

  return (
    <div className="flex min-h-screen bg-neutral-900">
      {/* Left Panel */}
      <div className="w-1/3 bg-neutral-950 shadow-md px-8 pt-6 pb-8 flex flex-col">
        <h1 className="text-2xl font-bold mb-6 text-neutral-200">
          Multi-Stock LSTM Predictor
        </h1>

        {error && (
          <div className="bg-red-500 text-white p-3 rounded mb-4">{error}</div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Stock Selection */}
          <div>
            <label className="block text-neutral-200 text-sm font-bold mb-2">
              Select Stocks ({selectedTickers.length}/8):
            </label>
            <div className="grid grid-cols-3 gap-2 max-h-48 overflow-y-auto">
              {popularTickers.map((ticker) => (
                <button
                  key={ticker}
                  type="button"
                  onClick={() => handleTickerToggle(ticker)}
                  className={`p-2 text-xs rounded border ${
                    selectedTickers.includes(ticker)
                      ? "bg-blue-600 text-white border-blue-600"
                      : "bg-neutral-800 text-neutral-300 border-neutral-600 hover:bg-neutral-700"
                  }`}>
                  {ticker}
                </button>
              ))}
            </div>
            {selectedTickers.length > 0 && (
              <div className="mt-2">
                <p className="text-neutral-400 text-sm">
                  Selected: {selectedTickers.join(", ")}
                </p>
              </div>
            )}
          </div>

          {/* Start Date */}
          <div>
            <label
              htmlFor="start-date"
              className="block text-neutral-200 text-sm font-bold mb-2">
              Start Date:
            </label>
            <input
              id="start-date"
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="appearance-none border-b-2 border-neutral-200 w-full py-2 px-3 text-neutral-200 leading-tight focus:outline-none focus:border-neutral-500 bg-transparent"
              required
            />
          </div>

          {/* End Date */}
          <div>
            <label
              htmlFor="end-date"
              className="block text-neutral-200 text-sm font-bold mb-2">
              End Date:
            </label>
            <input
              id="end-date"
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="appearance-none border-b-2 border-neutral-200 w-full py-2 px-3 text-neutral-200 leading-tight focus:outline-none focus:border-neutral-500 bg-transparent"
              required
            />
          </div>

          {/* Sequence Length */}
          <div>
            <label
              htmlFor="sequence-length"
              className="block text-neutral-200 text-sm font-bold mb-2 mt-8">
              Sequence Length: {sequenceLength}
            </label>
            <input
              id="sequence-length"
              type="range"
              min="20"
              max="40"
              step="5"
              value={sequenceLength}
              onChange={(e) => setSequenceLength(parseInt(e.target.value))}
              className="w-full"
            />
          </div>

          {/* Submit Button */}
          <div>
            <button
              type="submit"
              disabled={selectedTickers.length === 0}
              className="bg-neutral-300 hover:bg-neutral-100 disabled:bg-neutral-700 disabled:text-neutral-500 mt-8 text-black font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline w-full">
              {selectedTickers.length === 0
                ? "Select Stocks First"
                : "Generate Predictions"}
            </button>
          </div>
        </form>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col p-8">
        {loading ? (
          <div className="text-center">
            <h1 className="mt-4 text-neutral-200 font-bold text-lg">
              {loading}
            </h1>
          </div>
        ) : plotUrls.length > 0 ? (
          <div className="w-full">
            <h2 className="text-4xl font-bold mb-6 text-neutral-200">
              Results
            </h2>

            {/* Combined Plots and Metrics */}
            {plotUrls.length > 0 && Object.keys(metrics).length > 0 && (
              <div className="space-y-8">
                {plotUrls.map((url, index) => {
                  // Extract ticker from the plot URL (e.g., "/static/AAPL_results.png" -> "AAPL")
                  const ticker = url.split("/").pop().split("_")[0];
                  const tickerMetrics = metrics[ticker];

                  if (!tickerMetrics) return null;

                  return (
                    <div key={index} className="w-full">
                      <div className="grid grid-cols-5 gap-4 items-stretch">
                        {/* Plot Section - Takes up 80% of the width (4/5 columns) */}
                        <div className="col-span-4">
                          <div className="bg-neutral-800 p-6 rounded-lg border border-neutral-700 h-full">
                            <img
                              src={`${REACT_APP_API_URL}${url}?t=${new Date().getTime()}`}
                              alt={`${ticker} Stock Prediction`}
                              className="w-full h-full object-contain"
                              style={{
                                border: "none",
                                backgroundColor: "transparent",
                                borderRadius: "8px",
                              }}
                            />
                          </div>
                        </div>

                        {/* Performance Metrics Section - Takes up 20% of the width (1/5 column) */}
                        <div className="col-span-1">
                          <div className="bg-neutral-800 p-4 rounded-lg border border-neutral-700 h-full flex flex-col justify-center">
                            <h4 className="text-lg font-bold text-neutral-200 mb-3 text-center">
                              {ticker}
                            </h4>
                            <div className="space-y-2 text-xs">
                              <p className="text-neutral-300">
                                Strategy Return:{" "}
                                <span
                                  className={`font-bold ${tickerMetrics.total_return >= 0 ? "text-green-400" : "text-red-400"}`}>
                                  {tickerMetrics.total_return.toFixed(2)}%
                                </span>
                              </p>
                              <p className="text-neutral-300">
                                Buy & Hold Return:{" "}
                                <span
                                  className={`font-bold ${tickerMetrics.market_return >= 0 ? "text-green-400" : "text-red-400"}`}>
                                  {tickerMetrics.market_return.toFixed(2)}%
                                </span>
                              </p>
                              <p className="text-neutral-300">
                                Sharpe Ratio:{" "}
                                <span className="font-bold text-blue-400">
                                  {tickerMetrics.sharpe_ratio.toFixed(2)}
                                </span>
                              </p>
                              <p className="text-neutral-300">
                                Max Drawdown:{" "}
                                <span className="font-bold text-red-400">
                                  {tickerMetrics.max_drawdown.toFixed(2)}%
                                </span>
                              </p>
                              <p className="text-neutral-300">
                                Final Value:{" "}
                                <span className="font-bold text-green-400">
                                  $
                                  {Math.round(
                                    tickerMetrics.final_value
                                  ).toLocaleString()}
                                </span>
                              </p>

                              {/* Performance Summary */}
                              <div className="mt-2 pt-2 border-t border-neutral-600">
                                <p className="text-neutral-400 text-xs">
                                  vs Buy & Hold:{" "}
                                  <span
                                    className={`font-bold ${tickerMetrics.total_return - tickerMetrics.market_return >= 0 ? "text-green-400" : "text-red-400"}`}>
                                    {tickerMetrics.total_return >
                                    tickerMetrics.market_return
                                      ? "+"
                                      : ""}
                                    {(
                                      tickerMetrics.total_return -
                                      tickerMetrics.market_return
                                    ).toFixed(1)}
                                    %
                                  </span>
                                </p>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        ) : (
          <div className="flex items-center justify-center h-full">
            <h2 className="text-xl font-semibold text-gray-500">
              Please select stocks and submit to see predictions.
            </h2>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
