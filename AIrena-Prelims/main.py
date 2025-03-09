import argparse
import random
import time
from agent1 import CustomPlayer
from agent2 import MinimaxPlayer
from controller import Controller
from gui import GUI

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def run_single_game(size, player1_type, player2_type):
    player_classes = {"Agent": CustomPlayer, "minimax": MinimaxPlayer}
    player1 = player_classes[player1_type](size, 1, 2)
    player2 = player_classes[player2_type](size, 2, 1)
    
    controller = Controller(size, player1, player2)
    
    while controller._winner == 0:
        controller.update()
    
    return {
        "winner": controller._winner,
        "winner_name": player1.name if controller._winner == 1 else player2.name,
        "player1_type": player1_type,
        "player2_type": player2_type
    }

def run_evaluation(num_games=25, size=5):
    print(f"\n{'=' * 60}")
    print(f"EVALUATION MODE: Running {num_games} games with board size {size}")
    print(f"{'=' * 60}\n")
    
    results = []
    player_types = ["Agent", "minimax"]
    stats = {
        "Agent_wins": 0,
        "minimax_wins": 0,
        "Agent_as_p1_wins": 0,
        "minimax_as_p1_wins": 0,
        "Agent_as_p2_wins": 0,
        "minimax_as_p2_wins": 0,
        "total_games": num_games
    }
    
    start_time = time.time()
    
    for i in range(num_games):
        # Randomly assign player types to positions
        random.shuffle(player_types)
        player1_type, player2_type = player_types
        
        # print(f"Game {i+1}/{num_games}: Player 1 = {player1_type.capitalize()}, Player 2 = {player2_type.capitalize()}")
        
        result = run_single_game(size, player1_type, player2_type)
        results.append(result)
        
        # Update statistics
        winner_type = result["player1_type"] if result["winner"] == 1 else result["player2_type"]
        if winner_type == "Agent":
            stats["Agent_wins"] += 1
            if result["winner"] == 1:
                stats["Agent_as_p1_wins"] += 1
            else:
                stats["Agent_as_p2_wins"] += 1
        else:  # minimax
            stats["minimax_wins"] += 1
            if result["winner"] == 1:
                stats["minimax_as_p1_wins"] += 1
            else:
                stats["minimax_as_p2_wins"] += 1
                
        print(f"  → Winner: Player {result['winner']} ({result['winner_name']})\n")
    
    elapsed_time = time.time() - start_time
    
    # Calculate win percentages
    Agent_win_pct = (stats["Agent_wins"] / stats["total_games"]) * 100
    minimax_win_pct = (stats["minimax_wins"] / stats["total_games"]) * 100
    
    # Calculate win percentages as Player 1 vs Player 2
    Agent_games_as_p1 = sum(1 for r in results if r["player1_type"] == "Agent")
    Agent_p1_win_pct = (stats["Agent_as_p1_wins"] / Agent_games_as_p1) * 100 if Agent_games_as_p1 > 0 else 0
    
    Agent_games_as_p2 = sum(1 for r in results if r["player2_type"] == "Agent")
    Agent_p2_win_pct = (stats["Agent_as_p2_wins"] / Agent_games_as_p2) * 100 if Agent_games_as_p2 > 0 else 0
    
    minimax_games_as_p1 = sum(1 for r in results if r["player1_type"] == "minimax")
    minimax_p1_win_pct = (stats["minimax_as_p1_wins"] / minimax_games_as_p1) * 100 if minimax_games_as_p1 > 0 else 0
    
    minimax_games_as_p2 = sum(1 for r in results if r["player2_type"] == "minimax")
    minimax_p2_win_pct = (stats["minimax_as_p2_wins"] / minimax_games_as_p2) * 100 if minimax_games_as_p2 > 0 else 0
    
    # Print detailed statistics
    print(f"\n{'=' * 60}")
    print(f"EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"Total games played: {stats['total_games']}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average time per game: {elapsed_time / stats['total_games']:.2f} seconds")
    print(f"\nWin Statistics:")
    print(f"  Agent Agent: {stats['Agent_wins']} wins ({Agent_win_pct:.1f}%)")
    print(f"  Minimax Agent: {stats['minimax_wins']} wins ({minimax_win_pct:.1f}%)")
    print(f"\nPosition Analysis:")
    print(f"  Agent as Player 1 (top-bottom): {stats['Agent_as_p1_wins']}/{Agent_games_as_p1} wins ({Agent_p1_win_pct:.1f}%)")
    print(f"  Agent as Player 2 (left-right): {stats['Agent_as_p2_wins']}/{Agent_games_as_p2} wins ({Agent_p2_win_pct:.1f}%)")
    print(f"  Minimax as Player 1 (top-bottom): {stats['minimax_as_p1_wins']}/{minimax_games_as_p1} wins ({minimax_p1_win_pct:.1f}%)")
    print(f"  Minimax as Player 2 (left-right): {stats['minimax_as_p2_wins']}/{minimax_games_as_p2} wins ({minimax_p2_win_pct:.1f}%)")
    print(f"{'=' * 60}\n")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="AI-Rena Game")
    parser.add_argument("--gui", type=str2bool, default=True, help="Enable GUI (true/false)")
    parser.add_argument("--players", nargs=2, choices=["Agent", "minimax"], required=False, help="Player types")
    parser.add_argument("--size", type=int, default=9, help="Grid size")
    parser.add_argument("--eval", action="store_true", help="Run evaluation mode")
    args = parser.parse_args()
    
    # If eval mode is enabled, run the evaluation and exit
    if args.eval:
        run_evaluation(num_games=25, size=5)
        return
    
    # Regular game mode
    if not args.players:
        parser.error("the --players argument is required for regular game mode")
    
    player_classes = {"Agent": CustomPlayer, "minimax": MinimaxPlayer}
    player1 = player_classes[args.players[0]](args.size, 1, 2)
    player2 = player_classes[args.players[1]](args.size, 2, 1)

    controller = Controller(args.size, player1, player2)
    if args.gui:
        gui = GUI(controller)
        gui.start()
        gui.run()
    else:
        while controller._winner == 0:
            controller.update()
        print(f"Player {controller._winner} wins!")

if __name__ == "__main__":
    main()