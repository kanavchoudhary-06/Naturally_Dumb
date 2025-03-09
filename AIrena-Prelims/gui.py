import tkinter as tk
import numpy as np
from tkinter import font

class GUI:
    def __init__(self, controller):
        self._controller = controller
        self._grid = self._controller._grid
        self._screen = tk.Tk()
        self._screen.title("AI-Rena")
        self._screen.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Use a more modern color scheme
        self._bg_color = '#2C3E50'
        self._canvas_bg = '#ECF0F1'
        self._screen.configure(bg=self._bg_color)
        
        # Player colors
        self._p1_color = '#E74C3C'  # Red player
        self._p1_dark = '#C0392B'   # Dark red
        self._p2_color = '#3498DB'  # Blue player
        self._p2_dark = '#2980B9'   # Dark blue
        
        # Create a header with game title
        self._header_frame = tk.Frame(self._screen, bg=self._bg_color, pady=10)
        self._header_frame.pack(fill=tk.X)
        
        title_font = font.Font(family='Helvetica', size=24, weight='bold')
        tk.Label(self._header_frame, text="AI-Rena", fg=self._p1_color, bg=self._bg_color, 
                 font=title_font).pack()
        
        # Create main content frame
        content_frame = tk.Frame(self._screen, bg=self._bg_color)
        content_frame.pack(padx=20, pady=10, expand=True, fill=tk.BOTH)

        # Add game description
        description = "Connect your edges by capturing hexagons"
        tk.Label(content_frame, text=description, fg='#ECF0F1', bg=self._bg_color, 
                 font=('Helvetica', 11)).pack(pady=(0, 10))
        
        # Calculate canvas size with some padding
        hex_size = 22
        self._hex_size = hex_size
        width = int(np.sqrt(3) * hex_size * self._grid.get_size() * 1.8)  # Wider for border
        height = int(2 * hex_size * self._grid.get_size() * 1.3)  # Taller for border
        
        self._grid_canvas = tk.Canvas(content_frame, width=width, height=height, 
                                    bg=self._canvas_bg, bd=0, highlightthickness=0)
        self._grid_canvas.pack(pady=10)
        
        # Frame for player information
        self._info_frame = tk.Frame(content_frame, bg=self._bg_color, pady=10)
        self._info_frame.pack()
        
        # Player indicators with colored boxes
        p1_frame = tk.Frame(self._info_frame, bg=self._bg_color)
        p1_frame.pack(side=tk.LEFT, padx=20)
        
        p1_color_box = tk.Frame(p1_frame, width=20, height=20, bg=self._p1_color)
        p1_color_box.pack(side=tk.LEFT, padx=(0, 5))
        tk.Label(p1_frame, text="Player 1", fg=self._p1_color, bg=self._bg_color, 
                font=('Helvetica', 12, 'bold')).pack(side=tk.LEFT)
        tk.Label(p1_frame, text="(Left-Right)", fg='#ECF0F1', bg=self._bg_color, 
                font=('Helvetica', 10)).pack(side=tk.LEFT, padx=(5, 0))
        
        p2_frame = tk.Frame(self._info_frame, bg=self._bg_color)
        p2_frame.pack(side=tk.LEFT, padx=20)
        
        p2_color_box = tk.Frame(p2_frame, width=20, height=20, bg=self._p2_color)
        p2_color_box.pack(side=tk.LEFT, padx=(0, 5))
        tk.Label(p2_frame, text="Player 2", fg=self._p2_color, bg=self._bg_color, 
                font=('Helvetica', 12, 'bold')).pack(side=tk.LEFT)
        tk.Label(p2_frame, text="(Top-Bottom)", fg='#ECF0F1', bg=self._bg_color, 
                font=('Helvetica', 10)).pack(side=tk.LEFT, padx=(5, 0))
        
        # Turn and winner indicator
        self._turn_label = tk.Label(content_frame, text="", fg='#ECF0F1', bg=self._bg_color,
                                   font=('Helvetica', 12))
        self._turn_label.pack(pady=5)
        
        # Winner announcement label (initially hidden)
        self._winner_label = tk.Label(content_frame, text="", fg='#ECF0F1', bg=self._bg_color,
                                    font=('Helvetica', 18, 'bold'))
        self._winner_label.pack(pady=5)
        
        # Control buttons frame
        control_frame = tk.Frame(content_frame, bg=self._bg_color)
        control_frame.pack(pady=10)
        
        # Styled buttons
        button_style = {'font': ('Helvetica', 12), 'borderwidth': 0, 
                        'pady': 8, 'padx': 15, 'border': 0}
        
        self._start_button = tk.Button(control_frame, text="Start Game", bg='#2ECC71', fg='white',
                                     activebackground='#27AE60', activeforeground='white', 
                                     command=self.start, **button_style)
        self._start_button.pack(side=tk.LEFT, padx=10)
        
        quit_button = tk.Button(control_frame, text="Quit", bg='#E74C3C', fg='white',
                              activebackground='#C0392B', activeforeground='white',
                              command=self.on_closing, **button_style)
        quit_button.pack(side=tk.LEFT, padx=10)
        
        # Status bar
        self._status_frame = tk.Frame(self._screen, bg='#34495E', height=25)
        self._status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self._status_label = tk.Label(self._status_frame, text="Ready to play", 
                                    fg='#ECF0F1', bg='#34495E', anchor='w', padx=10)
        self._status_label.pack(fill=tk.X)
        
        self.stop = False
        self.launch = False
        self._draw()
        self._update_turn_label()

    def _update_turn_label(self):
        if not self.launch:
            self._turn_label.config(text="")
            return
        
        if self._controller._winner != 0:
            self._turn_label.config(text="")
            return
            
        current = self._controller._current_player
        color = self._p1_color if current == 1 else self._p2_color
        self._turn_label.config(text=f"Player {current}'s Turn", fg=color)

    def run(self):
        while not self.stop:
            if self.launch:
                self._controller.update()
                self._draw()
                self._update_turn_label()
                self._screen.update()
                self.check_win()
            else:
                self._screen.update()

    def check_win(self):
        if self._controller._winner != 0:
            self.launch = False
            self._show_winner()

    def _draw(self):
        self._grid_canvas.delete("all")
        size = self._hex_size
        w = np.sqrt(3) * size
        h = 2 * size
        grid_size = self._grid.get_size()
        
        # Draw the hexagons for the game grid only - no borders
        for y in range(grid_size):
            for x in range(grid_size):
                pos_x = w/2 + (y*w/2) + x*w + w/2  # Centered in canvas
                pos_y = h/2 + (h*3/4)*y + h/2      # Centered in canvas
                
                player = self._grid.get_hex([x, y])
                if player == 1:
                    color = self._p1_color
                    outline = self._p1_dark
                elif player == 2:
                    color = self._p2_color
                    outline = self._p2_dark
                else:
                    color = '#ECF0F1'  # Empty cell
                    outline = '#BDC3C7'
                
                # Draw with shadow effect for a 3D look
                self._draw_hex([pos_x, pos_y], color, outline, size)

    def _draw_border_areas(self, size, w, h, grid_size):
        # Calculate offsets based on the main grid positioning
        x_offset = w
        y_offset = h
        
        # Top border (Player 1) - above the main grid
        for x in range(1, grid_size+1):
            pos_y = h/2
            pos_x = w/2 + x*w
            self._draw_hex([pos_x, pos_y], self._p1_dark, self._p1_dark, size)
        
        # Bottom border (Player 1) - below the main grid
        for x in range(1, grid_size+1):
            pos_y = h/2 + (h*3/4)*grid_size + h/2
            pos_x = w/2 + (grid_size-1)*w/2 + x*w
            self._draw_hex([pos_x, pos_y], self._p1_dark, self._p1_dark, size)
        
        # Left border (Player 2) - left of the main grid
        for y in range(grid_size):
            pos_x = w/2 + (y*w/2)
            pos_y = h/2 + (h*3/4)*y + h
            self._draw_hex([pos_x, pos_y], self._p2_dark, self._p2_dark, size)
        
        # Right border (Player 2) - right of the main grid
        for y in range(grid_size):
            pos_x = w/2 + (y*w/2) + (grid_size)*w + w
            pos_y = h/2 + (h*3/4)*y + h
            self._draw_hex([pos_x, pos_y], self._p2_dark, self._p2_dark, size)

    def _draw_hex(self, coordinates, color, outline, size):
        points = []
        for i in range(6):
            angle_deg = 60 * i - 30
            angle_rad = np.pi / 180 * angle_deg
            points.append(coordinates[0] + size * np.cos(angle_rad))
            points.append(coordinates[1] + size * np.sin(angle_rad))
        
        # Draw with slight 3D effect
        # First draw a darker shadow slightly offset
        shadow_points = []
        shadow_offset = 2
        for i in range(0, len(points), 2):
            shadow_points.append(points[i])
            shadow_points.append(points[i+1] + shadow_offset)
        
        darker_color = self._darken_color(color, 0.8)
        self._grid_canvas.create_polygon(shadow_points, fill=darker_color, outline=darker_color)
        
        # Then draw the main hexagon
        self._grid_canvas.create_polygon(points, fill=color, outline=outline, width=1)

    def _darken_color(self, hex_color, factor=0.7):
        """Darken a hex color by the given factor"""
        # Convert hex to RGB
        h = hex_color.lstrip('#')
        rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        
        # Darken
        rgb = tuple(int(c * factor) for c in rgb)
        
        # Convert back to hex
        return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'

    def _show_winner(self):
        winner = self._controller._winner
        player_name = self._controller._player1.name if winner == 1 else self._controller._player2.name
        color = self._p1_color if winner == 1 else self._p2_color
        
        # Update winner announcement (no popup)
        self._winner_label.config(
            text=f"{player_name} (Player {winner}) wins!", 
            fg=color
        )
        
        # Update status
        self._status_label.config(text=f"Game ended. Player {winner} wins!")
        
        # Change start button text
        self._start_button.config(text="Game Over")

    def start(self):
        self.launch = True
        self._start_button.config(text="Game in Progress")
        self._status_label.config(text="Game started")
        self._winner_label.config(text="")  # Clear any previous winner message
        self._update_turn_label()

    def on_closing(self):
        self.stop = True
        self._screen.destroy()
