"""
Main GUI application for Portfolio Factsheet Generator.
Simple Tkinter interface with 3-step workflow.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import logging
from datetime import datetime
import os
import sys
import traceback
import pandas as pd

# Add modules directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    GUI_WIDTH, GUI_HEIGHT, GUI_TITLE, GUI_FONT, GUI_BG_COLOR,
    PROJECT_ROOT, REPORTS_DIR
)
from modules.data_loader import PortfolioData, load_portfolio_data
from modules.data_fetcher import DataFetcher
from modules.sector_mapper import SectorMapper
from modules.portfolio_calc import PortfolioCalculator
from modules.report_gen import ReportGenerator


class PortfolioFactsheetApp:
    """Main GUI application for portfolio factsheet generation."""
    
    def __init__(self, root):
        self.root = root
        self.root.title(GUI_TITLE)
        self.root.geometry(f"{GUI_WIDTH}x{GUI_HEIGHT}")
        self.root.configure(bg=GUI_BG_COLOR)
        
        # Initialize components
        self.portfolio_data = None
        self.data_fetcher = None
        self.sector_mapper = None
        self.calculator = None
        self.report_generator = None
        
        # Queue for thread communication
        self.message_queue = queue.Queue()
        
        # Current step
        self.current_step = 1
        
        # Setup GUI
        self._setup_gui()
        
        # Start checking for messages
        self._check_queue()
        
        # Set up logging to GUI
        self._setup_logging()
    
    def _setup_gui(self):
        """Setup the GUI components."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="Portfolio Factsheet Generator",
            font=(GUI_FONT[0], 16, "bold")
        )
        title_label.grid(row=0, column=0, pady=(0, 20))
        
        # Step indicator
        self.step_frame = ttk.LabelFrame(main_frame, text="Step Progress", padding="10")
        self.step_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        
        self.step_labels = []
        for i in range(3):
            step_text = ["1. Load CSV File", "2. Resolve Missing Data", "3. Generate Report"][i]
            label = ttk.Label(self.step_frame, text=step_text, font=GUI_FONT)
            label.grid(row=0, column=i, padx=20)
            self.step_labels.append(label)
        
        # Main content area - make it expandable
        self.content_frame = ttk.Frame(main_frame)
        self.content_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        self.content_frame.columnconfigure(0, weight=1)
        self.content_frame.rowconfigure(0, weight=1)  # Allow content to expand
        
        # Status area - make it smaller
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)
        
        self.status_text = scrolledtext.ScrolledText(
            status_frame, 
            height=6,  # Reduced from 8
            font=("Courier", 9)
        )
        self.status_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Button frame - ensure it's always visible
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, sticky=(tk.E, tk.S), pady=(5, 0))
        
        self.next_button = ttk.Button(
            button_frame, 
            text="Next",
            command=self._next_step,
            state=tk.DISABLED
        )
        self.next_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        self.back_button = ttk.Button(
            button_frame,
            text="Back",
            command=self._previous_step,
            state=tk.DISABLED
        )
        self.back_button.pack(side=tk.RIGHT)
        
        # Initialize step 1
        self._show_step_1()
    
    def _setup_logging(self):
        """Setup logging to redirect to GUI status area."""
        class GUILogHandler(logging.Handler):
            def __init__(self, text_widget):
                super().__init__()
                self.text_widget = text_widget
            
            def emit(self, record):
                msg = self.format(record)
                self.text_widget.insert(tk.END, msg + "\n")
                self.text_widget.see(tk.END)
        
        # Create handler
        handler = GUILogHandler(self.status_text)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        
        # Add to root logger
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)
    
    def _check_queue(self):
        """Check for messages from background threads."""
        try:
            while True:
                msg = self.message_queue.get_nowait()
                if msg == "LOAD_COMPLETE":
                    self._on_load_complete()
                elif msg == "FETCH_COMPLETE":
                    self._on_fetch_complete()
                elif msg == "REPORT_COMPLETE":
                    self._on_report_complete()
                elif msg.startswith("ERROR:"):
                    self._show_error(msg[6:])
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._check_queue)
    
    def _show_step_1(self):
        """Show step 1: Load CSV file."""
        self._clear_content_frame()
        
        # Update step indicator
        self._update_step_indicator(1)
        
        # Create a canvas with scrollbar for step 1 content
        canvas = tk.Canvas(self.content_frame)
        scrollbar = ttk.Scrollbar(self.content_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Step 1 content inside scrollable frame
        content = ttk.Frame(scrollable_frame)
        content.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        content.columnconfigure(0, weight=1)
        
        # Instructions
        instructions = ttk.Label(
            content,
            text="Select your portfolio CSV file to begin.\n\nThe file should contain monthly portfolio data with columns:\n기준일, 종목코드, 종목명, 수량, 현재가, 비중(%), 섹터, 국가, 통화, 환율",
            font=GUI_FONT,
            justify=tk.LEFT
        )
        instructions.grid(row=0, column=0, pady=(0, 20), sticky=tk.W)
        
        # File selection
        file_frame = ttk.Frame(content)
        file_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        file_frame.columnconfigure(1, weight=1)
        
        ttk.Label(file_frame, text="CSV File:", font=GUI_FONT).grid(row=0, column=0, padx=(0, 10))
        
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, font=GUI_FONT)
        file_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        browse_button = ttk.Button(
            file_frame,
            text="Browse...",
            command=self._browse_file
        )
        browse_button.grid(row=0, column=2)
        
        # Load button
        self.load_button = ttk.Button(
            content,
            text="Load Portfolio Data",
            command=self._load_portfolio_data,
            state=tk.DISABLED
        )
        self.load_button.grid(row=2, column=0, pady=(10, 0))
        
        # Enable/disable load button based on file selection
        def update_load_button(*args):
            if self.file_path_var.get() and os.path.exists(self.file_path_var.get()):
                self.load_button.config(state=tk.NORMAL)
            else:
                self.load_button.config(state=tk.DISABLED)
        
        self.file_path_var.trace("w", update_load_button)
    
    def _show_step_2(self):
        """Show step 2: Resolve missing data."""
        self._clear_content_frame()
        
        # Update step indicator
        self._update_step_indicator(2)
        
        # Create a canvas with scrollbar for step 2 content
        canvas = tk.Canvas(self.content_frame)
        scrollbar = ttk.Scrollbar(self.content_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Step 2 content inside scrollable frame
        content = ttk.Frame(scrollable_frame)
        content.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        content.columnconfigure(0, weight=1)
        
        if not self.portfolio_data:
            self._show_error("No portfolio data loaded")
            return
        
        # Get missing data summary
        missing_summary = self.portfolio_data.get_missing_data_summary()
        
        # Display missing data information
        missing_text = f"Missing Data Found:\n\n"
        missing_text += f"• Stock Prices: {missing_summary['by_type'].get('prices', 0)}\n"
        missing_text += f"• Exchange Rates: {missing_summary['by_type'].get('exchange_rates', 0)}\n"
        missing_text += f"• Sectors: {missing_summary['by_type'].get('sectors', 0)}\n"
        missing_text += f"• Weights: {missing_summary['by_type'].get('weights', 0)}\n\n"
        missing_text += f"Total missing items: {missing_summary['total_missing']}"
        
        missing_label = ttk.Label(
            content,
            text=missing_text,
            font=GUI_FONT,
            justify=tk.LEFT
        )
        missing_label.grid(row=0, column=0, pady=(0, 20), sticky=tk.W)
        
        # Resolution options
        options_frame = ttk.LabelFrame(content, text="Resolution Options", padding="10")
        options_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        options_frame.columnconfigure(0, weight=1)
        
        self.resolution_var = tk.StringVar(value="auto")
        
        ttk.Radiobutton(
            options_frame,
            text="Auto-fetch missing data (Recommended)",
            variable=self.resolution_var,
            value="auto"
        ).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        ttk.Radiobutton(
            options_frame,
            text="Manual input for all missing data",
            variable=self.resolution_var,
            value="manual"
        ).grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        
        ttk.Radiobutton(
            options_frame,
            text="Use estimates (fill with available data)",
            variable=self.resolution_var,
            value="estimate"
        ).grid(row=2, column=0, sticky=tk.W)
        
        # Resolve button
        self.resolve_button = ttk.Button(
            content,
            text="Resolve Missing Data",
            command=self._resolve_missing_data
        )
        self.resolve_button.grid(row=2, column=0, pady=(20, 10))
        
        # Enable back button
        self.back_button.config(state=tk.NORMAL)
    
    def _show_step_3(self):
        """Show step 3: Generate report."""
        self._clear_content_frame()
        
        # Update step indicator
        self._update_step_indicator(3)
        
        # Create a canvas with scrollbar for step 3 content
        canvas = tk.Canvas(self.content_frame)
        scrollbar = ttk.Scrollbar(self.content_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Step 3 content inside scrollable frame
        content = ttk.Frame(scrollable_frame)
        content.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        content.columnconfigure(0, weight=1)
        
        # Portfolio summary
        if self.portfolio_data:
            summary = self.portfolio_data.get_portfolio_summary()
            
            summary_text = f"Portfolio Summary:\n\n"
            summary_text += f"• Period: {summary['date_range']['start']} to {summary['date_range']['end']}\n"
            summary_text += f"• Months: {summary['total_months']}\n"
            summary_text += f"• Unique Stocks: {summary['unique_stocks']}\n"
            
            # Safely handle countries list
            countries = summary.get('countries', [])
            country_list = []
            if countries:
                for c in countries:
                    # Check if value is valid (not None and not NaN)
                    if c is not None:
                        # Check for NaN (float NaN is not equal to itself)
                        if isinstance(c, float) and c != c:  # NaN check
                            continue
                        country_list.append(str(c))
            summary_text += f"• Countries: {', '.join(country_list) if country_list else 'N/A'}\n"
            
            # Safely handle currencies list
            currencies = summary.get('currencies_used', [])
            currency_list = []
            if currencies:
                for c in currencies:
                    # Check if value is valid (not None and not NaN)
                    if c is not None:
                        # Check for NaN (float NaN is not equal to itself)
                        if isinstance(c, float) and c != c:  # NaN check
                            continue
                        currency_list.append(str(c))
            summary_text += f"• Currencies: {', '.join(currency_list) if currency_list else 'N/A'}"
            
            summary_label = ttk.Label(
                content,
                text=summary_text,
                font=GUI_FONT,
                justify=tk.LEFT
            )
            summary_label.grid(row=0, column=0, pady=(0, 20), sticky=tk.W)
        
        # Report options
        options_frame = ttk.LabelFrame(content, text="Report Options", padding="10")
        options_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        options_frame.columnconfigure(0, weight=1)
        
        # Report title
        ttk.Label(options_frame, text="Report Title:", font=GUI_FONT).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.report_title_var = tk.StringVar(value="Portfolio Performance Factsheet")
        title_entry = ttk.Entry(options_frame, textvariable=self.report_title_var, font=GUI_FONT)
        title_entry.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Include charts
        self.include_charts_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame,
            text="Include charts in report",
            variable=self.include_charts_var
        ).grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        
        # Open after generation
        self.open_after_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame,
            text="Open report after generation",
            variable=self.open_after_var
        ).grid(row=3, column=0, sticky=tk.W)
        
        # Generate button
        self.generate_button = ttk.Button(
            content,
            text="Generate Report",
            command=self._generate_report
        )
        self.generate_button.grid(row=2, column=0, pady=(10, 0))
        
        # Enable back button, disable next button
        self.back_button.config(state=tk.NORMAL)
        self.next_button.config(state=tk.DISABLED)
    
    def _clear_content_frame(self):
        """Clear the content frame."""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
    
    def _update_step_indicator(self, step):
        """Update the step indicator."""
        self.current_step = step
        
        for i, label in enumerate(self.step_labels):
            if i + 1 == step:
                label.config(foreground="blue", font=(GUI_FONT[0], GUI_FONT[1], "bold"))
            elif i + 1 < step:
                label.config(foreground="green", font=GUI_FONT)
            else:
                label.config(foreground="gray", font=GUI_FONT)
        
        # Update button states
        if step == 1:
            self.back_button.config(state=tk.DISABLED)
            self.next_button.config(state=tk.DISABLED)
        elif step == 2:
            self.back_button.config(state=tk.NORMAL)
            self.next_button.config(state=tk.DISABLED)
        elif step == 3:
            self.back_button.config(state=tk.NORMAL)
            self.next_button.config(state=tk.DISABLED)
    
    def _browse_file(self):
        """Browse for CSV file."""
        filename = filedialog.askopenfilename(
            title="Select Portfolio CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.file_path_var.set(filename)
    
    def _load_portfolio_data(self):
        """Load portfolio data from CSV file."""
        filepath = self.file_path_var.get()
        
        if not filepath or not os.path.exists(filepath):
            self._show_error("Please select a valid CSV file")
            return
        
        # Disable button during loading
        self.load_button.config(state=tk.DISABLED, text="Loading...")
        self.status_text.delete(1.0, tk.END)
        
        # Load in background thread
        def load_thread():
            try:
                logging.info(f"Loading portfolio data from: {filepath}")
                portfolio, error = load_portfolio_data(filepath)
                
                if error:
                    self.message_queue.put(f"ERROR:{error}")
                else:
                    self.portfolio_data = portfolio
                    self.message_queue.put("LOAD_COMPLETE")
                    
            except Exception as e:
                self.message_queue.put(f"ERROR:Error loading portfolio data: {str(e)}")
        
        threading.Thread(target=load_thread, daemon=True).start()
    
    def _on_load_complete(self):
        """Handle completion of portfolio loading."""
        self.load_button.config(state=tk.NORMAL, text="Load Portfolio Data")
        
        if self.portfolio_data:
            # Show portfolio summary
            summary = self.portfolio_data.get_portfolio_summary()
            missing = self.portfolio_data.get_missing_data_summary()
            
            logging.info(f"Portfolio loaded: {summary['total_months']} months, {summary['unique_stocks']} stocks")
            logging.info(f"Missing data: {missing['total_missing']} items")
            
            # Enable next button
            self.next_button.config(state=tk.NORMAL)
        else:
            self._show_error("Failed to load portfolio data")
    
    def _resolve_missing_data(self):
        """Resolve missing data based on selected option."""
        resolution_method = self.resolution_var.get()
        
        # Disable button during processing
        self.resolve_button.config(state=tk.DISABLED, text="Processing...")
        self.status_text.delete(1.0, tk.END)
        
        # Process in background thread
        def resolve_thread():
            try:
                logging.info(f"Resolving missing data using method: {resolution_method}")
                
                # Initialize data fetcher and sector mapper
                self.data_fetcher = DataFetcher()
                self.sector_mapper = SectorMapper(self.data_fetcher)
                
                # Get missing data
                missing_data = self.portfolio_data.missing_data
                
                # Resolve based on method
                if resolution_method == "auto":
                    self._auto_fetch_missing_data(missing_data)
                elif resolution_method == "manual":
                    # Show manual input dialog
                    self.root.after(0, lambda: self._show_manual_input_dialog(missing_data))
                elif resolution_method == "estimate":
                    self._estimate_missing_data(missing_data)
                
            except Exception as e:
                self.message_queue.put(f"ERROR:Error resolving missing data: {str(e)}")
        
        threading.Thread(target=resolve_thread, daemon=True).start()
    
    def _auto_fetch_missing_data(self, missing_data):
        """Auto-fetch missing data using APIs."""
        updates = []
        fetched_count = 0
        failed_count = 0
        
        # Fetch missing prices
        price_items = missing_data.get("prices", [])
        logging.info(f"Fetching {len(price_items)} missing prices...")
        
        for item in price_items:
            logging.info(f"Fetching price for {item['ticker']} on {item['date']}")
            price = self.data_fetcher.fetch_stock_price(item['ticker'], item['date'])
            if price:
                updates.append({
                    "type": "prices",
                    "ticker": item['ticker'],
                    "date": item['date'],
                    "value": price
                })
                fetched_count += 1
                logging.info(f"Fetched price: {price}")
            else:
                failed_count += 1
                logging.warning(f"Failed to fetch price for {item['ticker']}")
        
        # Fetch missing exchange rates
        rate_items = missing_data.get("exchange_rates", [])
        logging.info(f"Fetching {len(rate_items)} missing exchange rates...")
        
        for item in rate_items:
            logging.info(f"Fetching exchange rate for {item['currency']} on {item['date']}")
            rate = self.data_fetcher.fetch_exchange_rate(item['currency'], "KRW", item['date'])
            if rate:
                updates.append({
                    "type": "exchange_rates",
                    "currency": item['currency'],
                    "date": item['date'],
                    "value": rate
                })
                fetched_count += 1
                logging.info(f"Fetched rate: {rate}")
            else:
                failed_count += 1
                logging.warning(f"Failed to fetch exchange rate for {item['currency']}")
        
        # Fetch missing sectors
        sector_items = missing_data.get("sectors", [])
        logging.info(f"Fetching {len(sector_items)} missing sectors...")
        
        for item in sector_items:
            ticker = item['ticker']
            
            # Handle benchmarks (set sector to "Benchmark")
            if ticker in ["KOSPI", "S&P", "SPX"]:
                sector = "Benchmark"
                logging.info(f"Setting benchmark sector for {ticker}: {sector}")
            else:
                logging.info(f"Fetching sector for {ticker}")
                sector = self.sector_mapper.get_sector_for_ticker(ticker)
            
            if sector:
                updates.append({
                    "type": "sectors",
                    "ticker": ticker,
                    "date": item['date'],
                    "value": sector
                })
                fetched_count += 1
                logging.info(f"Fetched sector: {sector}")
            else:
                failed_count += 1
                logging.warning(f"Failed to fetch sector for {ticker}")
        
        # Apply updates to portfolio data
        if updates:
            for update in updates:
                self.portfolio_data.update_missing_data(update["type"], [update])
            logging.info(f"Applied {len(updates)} updates to portfolio data")
        
        # Save cache
        self.data_fetcher.save_caches()
        
        # Calculate weights if needed
        if len(self.portfolio_data.missing_data["weights"]) > 0:
            self.portfolio_data.calculate_missing_weights()
        
        logging.info(f"Auto-fetch completed: {fetched_count} items fetched, {failed_count} failed")
        self.message_queue.put("FETCH_COMPLETE")
    
    def _estimate_missing_data(self, missing_data):
        """Estimate missing data using available information."""
        df = self.portfolio_data.processed_data
        
        updates = []
        
        # Estimate missing prices (use last available price)
        for item in missing_data.get("prices", []):
            ticker_data = df[df['ticker'] == item['ticker']]
            if not ticker_data.empty:
                # Find closest price
                available_prices = ticker_data[ticker_data['price'].notna()]
                if not available_prices.empty:
                    # Use average of available prices
                    estimated_price = available_prices['price'].mean()
                    updates.append({
                        "type": "prices",
                        "ticker": item['ticker'],
                        "date": item['date'],
                        "value": estimated_price
                    })
                    logging.info(f"Estimated price for {item['ticker']}: {estimated_price}")
        
        # Estimate missing exchange rates (use last available rate)
        for item in missing_data.get("exchange_rates", []):
            currency_data = df[df['currency'] == item['currency']]
            if not currency_data.empty:
                available_rates = currency_data[currency_data['exchange_rate'].notna()]
                if not available_rates.empty:
                    estimated_rate = available_rates['exchange_rate'].mean()
                    updates.append({
                        "type": "exchange_rates",
                        "currency": item['currency'],
                        "date": item['date'],
                        "value": estimated_rate
                    })
                    logging.info(f"Estimated rate for {item['currency']}: {estimated_rate}")
        
        # Apply updates
        for update in updates:
            self.portfolio_data.update_missing_data(update["type"], [update])
        
        logging.info(f"Estimation completed: {len(updates)} items estimated")
        self.message_queue.put("FETCH_COMPLETE")
    
    def _on_fetch_complete(self):
        """Handle completion of missing data resolution."""
        self.resolve_button.config(state=tk.NORMAL, text="Resolve Missing Data")
        
        # Calculate weights if needed
        if len(self.portfolio_data.missing_data["weights"]) > 0:
            self.portfolio_data.calculate_missing_weights()
        
        # Show updated missing data summary
        missing_summary = self.portfolio_data.get_missing_data_summary()
        logging.info(f"Resolution complete. Remaining missing: {missing_summary['total_missing']} items")
        
        # Ask user if they want to save the updated data
        self._ask_save_updated_data()
        
        # Enable next button
        self.next_button.config(state=tk.NORMAL)
    
    def _ask_save_updated_data(self):
        """Ask user if they want to save the updated data."""
        response = messagebox.askyesno(
            "Save Updated Data",
            "Do you want to save the updated portfolio data to a new CSV file?\n\n"
            "This will create a new file with all fetched and calculated values."
        )
        
        if response:
            self._save_updated_data()
    
    def _save_updated_data(self):
        """Save updated portfolio data to a new CSV file."""
        try:
            from config import UPDATED_DATA_DIR
            from datetime import datetime
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            original_filename = os.path.basename(self.file_path_var.get())
            filename_base = os.path.splitext(original_filename)[0]
            new_filename = f"{filename_base}_updated_{timestamp}.csv"
            save_path = os.path.join(UPDATED_DATA_DIR, new_filename)
            
            # Save data in original format
            success = self.portfolio_data.save_to_original_format(save_path)
            
            if success:
                logging.info(f"Saved updated data to: {save_path}")
                messagebox.showinfo(
                    "Data Saved",
                    f"Updated portfolio data has been saved to:\n\n{save_path}"
                )
            else:
                messagebox.showerror(
                    "Save Failed",
                    "Failed to save updated portfolio data."
                )
                
        except Exception as e:
            logging.error(f"Error saving updated data: {e}")
            messagebox.showerror(
                "Error",
                f"Error saving updated data: {str(e)}"
            )
    
    def _generate_report(self):
        """Generate the portfolio factsheet report."""
        # Disable button during generation
        self.generate_button.config(state=tk.DISABLED, text="Generating...")
        self.status_text.delete(1.0, tk.END)
        
        # Process in background thread
        def generate_thread():
            try:
                logging.info("Starting report generation...")
                
                # Calculate portfolio performance
                self.calculator = PortfolioCalculator()
                portfolio_df = self.portfolio_data.get_data_for_calculation()
                calculation_results = self.calculator.calculate_all(portfolio_df)
                
                if not calculation_results:
                    self.message_queue.put("ERROR:Failed to calculate portfolio performance")
                    return
                
                # Generate report
                self.report_generator = ReportGenerator()
                report_path = self.report_generator.generate_report(
                    portfolio_data=portfolio_df,
                    calculation_results=calculation_results,
                    title=self.report_title_var.get(),
                    include_charts=self.include_charts_var.get()
                )
                
                if report_path and os.path.exists(report_path):
                    logging.info(f"Report generated: {report_path}")
                    
                    # Open report if requested
                    if self.open_after_var.get():
                        import webbrowser
                        webbrowser.open(f"file://{report_path}")
                    
                    self.message_queue.put("REPORT_COMPLETE")
                else:
                    self.message_queue.put("ERROR:Failed to generate report")
                
            except Exception as e:
                error_msg = f"Error generating report: {str(e)}\n{traceback.format_exc()}"
                self.message_queue.put(f"ERROR:{error_msg}")
        
        threading.Thread(target=generate_thread, daemon=True).start()
    
    def _on_report_complete(self):
        """Handle completion of report generation."""
        self.generate_button.config(state=tk.NORMAL, text="Generate Report")
        logging.info("Report generation completed successfully!")
        
        # Show success message
        messagebox.showinfo(
            "Success",
            "Portfolio factsheet report has been generated successfully!\n\n"
            "The report has been saved to the outputs/reports directory."
        )
    
    def _next_step(self):
        """Move to the next step."""
        if self.current_step == 1:
            self._show_step_2()
        elif self.current_step == 2:
            self._show_step_3()
    
    def _previous_step(self):
        """Move to the previous step."""
        if self.current_step == 2:
            self._show_step_1()
        elif self.current_step == 3:
            self._show_step_2()
    
    def _show_error(self, message):
        """Show error message."""
        logging.error(message)
        messagebox.showerror("Error", message)
    
    def _show_manual_input_dialog(self, missing_data):
        """Show dialog for manual input of missing data."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Manual Data Input")
        dialog.geometry("800x600")
        dialog.configure(bg=GUI_BG_COLOR)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Main container
        main_frame = ttk.Frame(dialog, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="Manual Data Input - Enter Missing Values",
            font=(GUI_FONT[0], 12, "bold")
        )
        title_label.grid(row=0, column=0, pady=(0, 10))
        
        # Create notebook for different data types
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        
        # Store input widgets
        self.manual_input_widgets = {}
        
        # Create tabs for each data type
        if missing_data.get("prices"):
            self._create_price_input_tab(notebook, missing_data["prices"])
        
        if missing_data.get("exchange_rates"):
            self._create_exchange_rate_input_tab(notebook, missing_data["exchange_rates"])
        
        if missing_data.get("sectors"):
            self._create_sector_input_tab(notebook, missing_data["sectors"])
        
        if missing_data.get("weights"):
            self._create_weight_input_tab(notebook, missing_data["weights"])
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, sticky="e")
        
        cancel_button = ttk.Button(
            button_frame,
            text="Cancel",
            command=dialog.destroy
        )
        cancel_button.pack(side=tk.LEFT, padx=(0, 5))
        
        save_button = ttk.Button(
            button_frame,
            text="Save All Inputs",
            command=lambda: self._save_manual_inputs(dialog)
        )
        save_button.pack(side=tk.LEFT)
    
    def _create_price_input_tab(self, notebook, price_items):
        """Create tab for price input."""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=f"Stock Prices ({len(price_items)})")
        
        # Create scrollable frame
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Header
        ttk.Label(scrollable_frame, text="Ticker", font=(GUI_FONT[0], 10, "bold")).grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(scrollable_frame, text="Date", font=(GUI_FONT[0], 10, "bold")).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(scrollable_frame, text="Price", font=(GUI_FONT[0], 10, "bold")).grid(row=0, column=2, padx=5, pady=5)
        
        # Price input fields
        self.manual_input_widgets["prices"] = []
        for i, item in enumerate(price_items, 1):
            ttk.Label(scrollable_frame, text=item['ticker']).grid(row=i, column=0, padx=5, pady=2)
            ttk.Label(scrollable_frame, text=item['date'].strftime('%Y-%m-%d')).grid(row=i, column=1, padx=5, pady=2)
            
            price_var = tk.StringVar()
            price_entry = ttk.Entry(scrollable_frame, textvariable=price_var, width=15)
            price_entry.grid(row=i, column=2, padx=5, pady=2)
            
            self.manual_input_widgets["prices"].append({
                "item": item,
                "var": price_var,
                "widget": price_entry
            })
    
    def _create_exchange_rate_input_tab(self, notebook, rate_items):
        """Create tab for exchange rate input."""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=f"Exchange Rates ({len(rate_items)})")
        
        # Create scrollable frame
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Header
        ttk.Label(scrollable_frame, text="Currency", font=(GUI_FONT[0], 10, "bold")).grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(scrollable_frame, text="Date", font=(GUI_FONT[0], 10, "bold")).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(scrollable_frame, text="Rate to KRW", font=(GUI_FONT[0], 10, "bold")).grid(row=0, column=2, padx=5, pady=5)
        
        # Exchange rate input fields
        self.manual_input_widgets["exchange_rates"] = []
        for i, item in enumerate(rate_items, 1):
            ttk.Label(scrollable_frame, text=item['currency']).grid(row=i, column=0, padx=5, pady=2)
            ttk.Label(scrollable_frame, text=item['date'].strftime('%Y-%m-%d')).grid(row=i, column=1, padx=5, pady=2)
            
            rate_var = tk.StringVar()
            rate_entry = ttk.Entry(scrollable_frame, textvariable=rate_var, width=15)
            rate_entry.grid(row=i, column=2, padx=5, pady=2)
            
            self.manual_input_widgets["exchange_rates"].append({
                "item": item,
                "var": rate_var,
                "widget": rate_entry
            })
    
    def _create_sector_input_tab(self, notebook, sector_items):
        """Create tab for sector input."""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=f"Sectors ({len(sector_items)})")
        
        # Create scrollable frame
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Header
        ttk.Label(scrollable_frame, text="Ticker", font=(GUI_FONT[0], 10, "bold")).grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(scrollable_frame, text="Date", font=(GUI_FONT[0], 10, "bold")).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(scrollable_frame, text="Sector", font=(GUI_FONT[0], 10, "bold")).grid(row=0, column=2, padx=5, pady=5)
        
        # Sector input fields with dropdown
        from config import VALID_SECTORS
        
        self.manual_input_widgets["sectors"] = []
        for i, item in enumerate(sector_items, 1):
            ttk.Label(scrollable_frame, text=item['ticker']).grid(row=i, column=0, padx=5, pady=2)
            ttk.Label(scrollable_frame, text=item['date'].strftime('%Y-%m-%d')).grid(row=i, column=1, padx=5, pady=2)
            
            sector_var = tk.StringVar()
            sector_combo = ttk.Combobox(scrollable_frame, textvariable=sector_var, 
                                       values=VALID_SECTORS, width=20, state="readonly")
            sector_combo.grid(row=i, column=2, padx=5, pady=2)
            
            self.manual_input_widgets["sectors"].append({
                "item": item,
                "var": sector_var,
                "widget": sector_combo
            })
    
    def _create_weight_input_tab(self, notebook, weight_items):
        """Create tab for weight input."""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=f"Weights ({len(weight_items)})")
        
        # Create scrollable frame
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Header
        ttk.Label(scrollable_frame, text="Ticker", font=(GUI_FONT[0], 10, "bold")).grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(scrollable_frame, text="Date", font=(GUI_FONT[0], 10, "bold")).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(scrollable_frame, text="Weight (%)", font=(GUI_FONT[0], 10, "bold")).grid(row=0, column=2, padx=5, pady=5)
        
        # Weight input fields
        self.manual_input_widgets["weights"] = []
        for i, item in enumerate(weight_items, 1):
            ttk.Label(scrollable_frame, text=item['ticker']).grid(row=i, column=0, padx=5, pady=2)
            ttk.Label(scrollable_frame, text=item['date'].strftime('%Y-%m-%d')).grid(row=i, column=1, padx=5, pady=2)
            
            weight_var = tk.StringVar()
            weight_entry = ttk.Entry(scrollable_frame, textvariable=weight_var, width=15)
            weight_entry.grid(row=i, column=2, padx=5, pady=2)
            
            self.manual_input_widgets["weights"].append({
                "item": item,
                "var": weight_var,
                "widget": weight_entry
            })
    
    def _save_manual_inputs(self, dialog):
        """Save all manually entered data."""
        updates = []
        
        # Process price inputs
        for widget_info in self.manual_input_widgets.get("prices", []):
            value = widget_info["var"].get()
            if value:
                try:
                    price = float(value)
                    updates.append({
                        "type": "prices",
                        "ticker": widget_info["item"]["ticker"],
                        "date": widget_info["item"]["date"],
                        "value": price
                    })
                except ValueError:
                    messagebox.showerror("Invalid Input", 
                                       f"Invalid price for {widget_info['item']['ticker']}: {value}")
                    return
        
        # Process exchange rate inputs
        for widget_info in self.manual_input_widgets.get("exchange_rates", []):
            value = widget_info["var"].get()
            if value:
                try:
                    rate = float(value)
                    updates.append({
                        "type": "exchange_rates",
                        "currency": widget_info["item"]["currency"],
                        "date": widget_info["item"]["date"],
                        "value": rate
                    })
                except ValueError:
                    messagebox.showerror("Invalid Input",
                                       f"Invalid exchange rate for {widget_info['item']['currency']}: {value}")
                    return
        
        # Process sector inputs
        for widget_info in self.manual_input_widgets.get("sectors", []):
            value = widget_info["var"].get()
            if value:
                updates.append({
                    "type": "sectors",
                    "ticker": widget_info["item"]["ticker"],
                    "date": widget_info["item"]["date"],
                    "value": value
                })
        
        # Process weight inputs
        for widget_info in self.manual_input_widgets.get("weights", []):
            value = widget_info["var"].get()
            if value:
                try:
                    weight = float(value)
                    updates.append({
                        "type": "weights",
                        "ticker": widget_info["item"]["ticker"],
                        "date": widget_info["item"]["date"],
                        "value": weight
                    })
                except ValueError:
                    messagebox.showerror("Invalid Input",
                                       f"Invalid weight for {widget_info['item']['ticker']}: {value}")
                    return
        
        # Apply updates to portfolio data
        for update in updates:
            self.portfolio_data.update_missing_data(update["type"], [update])
        
        # Save cache for any fetched data
        if self.data_fetcher:
            self.data_fetcher.save_caches()
        
        # Calculate weights if needed
        if len(self.portfolio_data.missing_data["weights"]) > 0:
            self.portfolio_data.calculate_missing_weights()
        
        # Show summary
        missing_summary = self.portfolio_data.get_missing_data_summary()
        logging.info(f"Manual input completed. {len(updates)} items saved. Remaining missing: {missing_summary['total_missing']} items")
        
        # Close dialog and proceed
        dialog.destroy()
        self.message_queue.put("FETCH_COMPLETE")


def main():
    """Main entry point for the application."""
    root = tk.Tk()
    app = PortfolioFactsheetApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()