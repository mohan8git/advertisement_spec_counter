import asyncio, websockets, json, time

ws_url = "ws://127.0.0.1:8000/ws/dashboard/"


# async def command_receiver():
# 	counter = 0	
#	async with websockets.connect(ws_url) as websocket:
#		while counter < 1000:		
#			message = await websocket.send(json.dumps({'users_count': counter}))
#			counter += 1



#asyncio.get_event_loop().run_until_complete(command_receiver())

async def command_receiver():
	async with websockets.connect(ws_url) as websocket:
		counter = 0
		while counter < 100:
			print("websocket : ", counter)		
			await websocket.send(json.dumps({'users_count': counter}))
			# time.sleep(0)
			counter += 1

asyncio.get_event_loop().run_until_complete(command_receiver())

